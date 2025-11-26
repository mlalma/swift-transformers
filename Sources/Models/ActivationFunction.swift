import Foundation
import MLX
import MLXNN

/// Applies the xIELU activation function introduced in https://arxiv.org/abs/2411.13010.
/// xIELU (exponential Inverse Exponential Linear Unit) is a learnable activation function that adapts during training.
/// It uses separate alpha parameters for positive and negative regions.
/// Formula:
/// - For x > 0: `alpha_p * x^2 + beta * x`
/// - For x ≤ 0: `(exp(min(x, eps)) - 1 - x) * alpha_n + beta * x`
/// Where alpha_p and alpha_n are learnable parameters constrained to be positive.
final class XIELUActivation: Module {
    /// Learnable parameter for positive region
    var alphaP: MLXArray
    /// Learnable parameter for negative region
    var alphaN: MLXArray
    /// Fixed parameter
    let beta: Float
    /// Small epsilon for numerical stability
    let eps: Float
    
    /// Initialize xIELU activation.
    /// - Parameters:
    ///   - alphaPInit: Initial value for alpha_p (default: 0.8)
    ///   - alphaNInit: Initial value for alpha_n (default: 0.8)
    ///   - beta: Beta parameter (default: 0.5)
    ///   - eps: Small epsilon for numerical stability (default: -1e-6)
    ///   - dtype: Data type for parameters (default: .float32)
    init(
        alphaPInit: Float = 0.8,
        alphaNInit: Float = 0.8,
        beta: Float = 0.5,
        eps: Float = -1e-6,
        dtype: DType = .float32
    ) {
        // Initialize alpha_p: log(expm1(alpha_p_init))
        // This ensures alpha_p stays positive when transformed with softplus
        let alphaPValue = log(MLX.expm1(MLXArray(alphaPInit)))
        self.alphaP = alphaPValue.asType(dtype).reshaped([1])
        
        // Initialize alpha_n: log(expm1(alpha_n_init - beta))
        let alphaNValue = log(MLX.expm1(MLXArray(alphaNInit - beta)))
        self.alphaN = alphaNValue.asType(dtype).reshaped([1])
        
        self.beta = beta
        self.eps = eps
        
        super.init()
    }
    
    /// Forward pass of xIELU activation.
    /// - Parameter x: Input tensor
    /// - Returns: Activated output
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Transform learnable parameters to ensure positivity
        // alpha_p = softplus(self.alpha_p)
        let alphaPTransformed = softplus(alphaP)
        
        // alpha_n = beta + softplus(self.alpha_n)
        let alphaNTransformed = beta + softplus(alphaN)
        
        // For positive values: alpha_p * x^2 + beta * x
        let positiveRegion = alphaPTransformed * x * x + beta * x
        
        // For negative values: (expm1(min(x, eps)) - x) * alpha_n + beta * x
        let clampedX = MLX.minimum(x, eps)
        let negativeRegion = (MLX.expm1(clampedX) - x) * alphaNTransformed + beta * x
        
        // Combine using where condition (x > 0)
        return MLX.where(x .> 0, positiveRegion, negativeRegion)
    }
}

/// Implements different kind of activation functions as a layer based on the given config.
final class ActivationFunction : Module {
    let activationFn: (MLXArray) -> MLXArray
    
    /// Initializes the activation function or nil if the activation function is not known.
    /// - Parameter activationFunction: Activation function from model config.
    init?(activationFunction: String) {
        switch activationFunction.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) {
        case "gelu", "gelu_python":
            self.activationFn = gelu
        case "gelu_10":
            self.activationFn = { MLX.minimum(MLX.maximum(gelu($0), -10.0), 10.0) }
        case "gelu_new", "gelu_pytorch_tanh", "gelu_python_tanh", "gelu_accurate", "gelu_fast":
            self.activationFn = geluApproximate
        case "laplace":
            self.activationFn = { input in
                // Laplace activation with default parameters introduced in https://huggingface.co/papers/2209.10655
                // mu = 0.707107, sigma = 0.282095
                let mu: Float = 0.707107
                let sigma: Float = 0.282095
                let sqrt2: Float = 1.41421356237
                
                // Transform: 0.5 * (1.0 + erf((input - mu) / (sigma * sqrt(2.0))))
                let transformed = (input - mu) / (sigma * sqrt2)
                return 0.5 * (1.0 + MLX.erf(transformed))
            }
        case "leaky_relu":
            self.activationFn = { leakyRelu($0, negativeSlope: 0.01) }
        case "linear":
            self.activationFn = { $0 }
        case "mish":
            self.activationFn = mish
        case "quick_gelu":
            self.activationFn = geluFastApproximate
        case "relu":
            self.activationFn = relu
        case "relu2":
            self.activationFn = reluSquared
        case "relu6":
            self.activationFn = relu6
        case "sigmoid":
            self.activationFn = sigmoid
        case "silu", "swish":
            self.activationFn = silu
        case "tanh":
            self.activationFn = { tanh($0) }
        case "prelu":
            let PReLU = MLXNN.PReLU()
            self.activationFn = { PReLU($0) }
        case "xielu":
            let XIELU = XIELUActivation()
            self.activationFn = { XIELU($0) }
        default:
            ModelUtils.log("Could not initialize activation function \(activationFunction)")
            return nil
        }
    }
    
    /// Forward pass for execution activation function on an MLP layer.
    /// - Parameter input: Input for the activation function.
    /// - Returns: Output array after activation function.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        return activationFn(input)
    }
}
