import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from scipy.optimize import minimize
from scipy import stats
import numpy as np

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
#
#                      ! DO NOT MODIFY THESE FUNCTIONS !
def integer_linear(input, weight):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.linear(input, weight)
    else:
        output = F.linear(input.float(), weight.float())
        output = output.round().to(torch.int32)
    return output
def integer_conv2d(input, weight, stride, padding, dilation, groups):
    assert input.dtype == torch.int32
    assert weight.dtype == torch.int32

    if 'cpu' in input.device.type:
        output = F.conv2d(input, weight, None, stride, padding, dilation, groups)
    else:
        output = F.conv2d(input.float(), weight.float(), None, stride, padding, dilation, groups)
        output = output.round().to(torch.int32)
    return output
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def linear_quantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor,
                    N_bits: int, signed: bool = True) -> torch.Tensor:
    """
    linear uniform quantization for real tensor
    Args:
        input: torch.tensor
        scale: scale factor
        zero_point: zero point
        N_bits: bitwidth
        signed: flag to indicate signed ot unsigned quantization

    Returns:
        quantized_tensor: quantized tensor whose values are integers
    """

    if signed:
        q_min = -(2 ** (N_bits - 1))
        q_max = 2 ** (N_bits - 1) - 1
    else:
        q_min = 0
        q_max = 2 ** N_bits - 1
    quantized_tensor = torch.clamp(torch.round(input / scale) - zero_point, q_min, q_max)
    return quantized_tensor

def linear_dequantize(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    linear uniform de-quantization for quantized tensor
    Args:
        input: input quantized tensor
        scale: scale factor
        zero_point: zero point

    Returns:
        reconstructed_tensor: de-quantized tensor whose values are real
    """
    return (input.float() + zero_point.float()) * scale.float()



def get_scale(input, N_bits=8):
    """
    extract optimal scale based on statistics of the input tensor.
    Args:
        input: input real tensor
        N_bits: bitwidth
    Returns:
        scale optimal scale
    """
    assert N_bits in [2, 4, 8]
    z_typical = {'2bit': [0.311, 0.678], '4bit': [0.077, 1.013], '8bit': [0.032, 1.085]}
    z = z_typical[f'{N_bits}bit']
    c1, c2 = 1 / z[0], z[1] / z[0]
    # SAWB: optimal clip value alpha = c1 * std(x) - c2 * E[|x|]
    std_val = input.std().item()
    mean_abs = input.abs().mean().item()
    clip_val = c1 * std_val - c2 * mean_abs
    q_scale = torch.tensor(max(clip_val, 1e-8), dtype=torch.float32)
    return q_scale


def reset_scale_and_zero_point(input: torch.tensor, N_bits: int = 4, method: str = "sym"):
    """
    Args:
        input: input real tensor
        N_bits: bitwidth
        method: choose between sym, asym, SAWB, and heuristic
    Returns:
        scale factor , zero point
    """
    with torch.no_grad():
        if method == 'heuristic':
            # step_size = argmin_{step_size} (MSE[x, x_hat])
            zero_point = torch.tensor(0.)

            def mse_fn(alpha):
                a = float(alpha[0])
                if a <= 0:
                    return 1e10
                s = torch.tensor(a / (2 ** (N_bits - 1) - 1), dtype=torch.float32)
                q = linear_quantize(input, s, zero_point, N_bits, signed=True)
                dq = linear_dequantize(q, s, zero_point)
                return F.mse_loss(input, dq).item()

            x0 = [input.abs().max().item()]
            result = minimize(mse_fn, x0, method='Nelder-Mead')
            alpha = abs(float(result.x[0]))
            step_size = torch.tensor(alpha / (2 ** (N_bits - 1) - 1), dtype=torch.float32)

        elif method == 'SAWB':
            # Statistics-Aware Weight Binning: use pre-computed c1/c2 from get_scale
            clip_val = get_scale(input, N_bits)
            q_max = 2 ** (N_bits - 1) - 1
            step_size = clip_val / q_max
            zero_point = torch.tensor(0.)

        elif method == 'sym':
            # Symmetric: clip at max absolute value, zero_point = 0
            max_val = input.abs().max()
            q_max = 2 ** (N_bits - 1) - 1
            step_size = max_val / q_max
            zero_point = torch.tensor(0.)

        elif method == 'asym':
            # Asymmetric: full range [min, max], unsigned quantization
            min_val = input.min()
            max_val = input.max()
            q_max = 2 ** N_bits - 1
            step_size = (max_val - min_val) / q_max
            zero_point = torch.round(min_val / step_size)

        else:
            raise "didn't find quantization method."

    return step_size, zero_point



class _quantize_func_STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, N_bits, signed=True):
        """
        Args:
            ctx: a context object that can be used to stash information for backward computation
            input: torch.tensor
            scale: scale factor
            zero_point: zero point
            N_bits: bitwidth
            signed: flag to indicate signed ot unsigned quantization
        Returns:
            quantized_tensor: quantized tensor whose values are integers
        """
        ctx.scale = scale
        quantized_tensor = linear_quantize(input, scale, zero_point, N_bits, signed)
        return quantized_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # STE: approximate quantize-dequantize as identity → grad passes through
        # dequantize multiplies by scale, so divide here to cancel it out
        grad_input = grad_output / ctx.scale
        return grad_input, None, None, None, None

linear_quantize_STE = _quantize_func_STE.apply


def quantized_linear_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float):
    """
    integer only fully connected layer. 
    Note that you are only allowed to use <integer_linear> function!
    Args:
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """

    # Integer-only linear: sum(q_x * q_w) then rescale by s_x * s_w
    output = integer_linear(input, weight)          # int32
    output = output.float() * input_scale * weight_scale
    if bias is not None:
        # bias was quantized with scale = weight_scale * input_scale
        output = output + bias.float() * input_scale * weight_scale
    return output


def quantized_conv2d_function(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                              input_scale: torch.float, weight_scale: torch.float, stride,
                              padding, dilation, groups):
    """
    integer only fully connected layer
    Note that you are only allowed to use <integer_conv2d> function!
    Args:
        groups: number of groups
        stride: stride
        dilation: dilation
        padding: padding
        input: quantized input
        weight: quantized weight
        bias: quantized bias
        input_scale: input scaling factor
        weight_scale: weight scaling factor

    Returns:
        output: output feature
    """

    # Integer-only conv2d: sum(q_x * q_w) then rescale by s_x * s_w
    output = integer_conv2d(input, weight, stride, padding, dilation, groups)  # int32
    output = output.float() * input_scale * weight_scale
    if bias is not None:
        # bias shape: (out_channels,) → reshape for (batch, out_channels, H, W) broadcasting
        output = output + bias.float().view(1, -1, 1, 1) * input_scale * weight_scale
    return output

class Quantized_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Quantized_Linear, self).__init__(in_features, out_features, bias=bias)

        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None
        self.input_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.decay = .99

    def forward(self, input):
        if self.method == 'normal':
            # default floating point mode.
            return F.linear(input, self.weight, self.bias)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point = torch.tensor(0.)
            # compute quantized
            signed_act = getattr(self, "act_signed", False)

            quantized_weight = linear_quantize_STE(self.weight, self.weight_scale, zero_point, self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point, self.act_N_bits, signed_act)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point, 32).to(torch.int32)
            output = quantized_linear_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
                                               quantized_bias, self.input_scale, self.weight_scale)
            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point)
            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale, zero_point)
            simulated_output = F.linear(input_reconstructed, weight_reconstructed, self.bias)
            return output + simulated_output - simulated_output.detach()

    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_unsigned(input, self.act_N_bits)
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        self.weight_scale.data, _= reset_scale_and_zero_point(self.weight, self.weight_N_bits, self.method)


class Quantized_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Quantized_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                               dilation=dilation, groups=groups, bias=bias)
        self.method = 'normal'  # normal, sym, asym, SAWB,
        self.act_N_bits = None
        self.weight_N_bits = None
        self.input_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.decay = .99

    def forward(self, input):
        if self.method == 'normal':
            # default floating point mode.
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            # update scale and zero
            self.__reset_scale_and_zero__(input)
            zero_point = torch.tensor(0.)
            # compute quantized
            signed_act = getattr(self, "act_signed", False)

            quantized_weight = linear_quantize_STE(self.weight, self.weight_scale, zero_point, self.weight_N_bits,True)
            quantized_input = linear_quantize_STE(input, self.input_scale, zero_point, self.act_N_bits, signed_act)
            if self.bias is None:
                quantized_bias = None
            else:
                quantized_bias = linear_quantize_STE(self.bias, self.weight_scale * self.input_scale, zero_point, 32).to(torch.int32)
            output = quantized_conv2d_function(quantized_input.to(torch.int32), quantized_weight.to(torch.int32),
                                               quantized_bias, self.input_scale, self.weight_scale, self.stride,
                                               self.padding, self.dilation, self.groups)
            input_reconstructed = linear_dequantize(quantized_input, self.input_scale, zero_point)
            weight_reconstructed = linear_dequantize(quantized_weight, self.weight_scale, zero_point)
            simulated_output = F.conv2d(input_reconstructed, weight_reconstructed, self.bias, self.stride, self.padding,
                                        self.dilation, self.groups)
            return output + simulated_output - simulated_output.detach()

    def __reset_scale_and_zero__(self, input):
        """
        update scale factor and zero point
            Args:
                input: input feature
            Returns:
        """
        if self.training:
            input_scale_update, _ = reset_scale_unsigned(input, self.act_N_bits)
            self.input_scale.data -= (1 - self.decay) * (self.input_scale - input_scale_update)
        self.weight_scale.data, _ = reset_scale_and_zero_point(self.weight, self.weight_N_bits, self.method)

def reset_scale_unsigned(input: torch.tensor, N_bits: int = 4):
    with torch.no_grad():
        zero_point = torch.tensor(0.)
        step_size = torch.max(torch.abs(input)) / ((2**(N_bits))-1)
    return step_size, zero_point