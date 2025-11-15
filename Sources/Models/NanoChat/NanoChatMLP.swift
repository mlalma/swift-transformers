import Foundation
import MLX
import MLXNN

/// Multi-Layer Perceptron (MLP) for NanoChat model.
final class NanoChatMLP: Module {
    let config: NanoChatConfig
    let activationFn: ActivationFunction
    let fc1: Linear
    let fc2: Linear
    
    /// Initialize the MLP with the given configuration.
    /// - Parameter config: Configuration for the NanoChat model.
    init?(config: NanoChatConfig) {
        self.config = config
        
        // Initialize activation function based on config
        guard let activation = ActivationFunction(activationFunction: config.hiddenAct) else {
            ModelUtils.log("Unknown activation function: \(config.hiddenAct)")
            return nil
        }
        self.activationFn = activation
        
        // Initialize linear layers without bias
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        
        super.init()
    }
    
    /// Forward pass through the MLP.
    /// - Parameter hiddenStates: Input tensor.
    /// - Returns: Output tensor after two linear transformations and activation.
    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var hiddenStates = hiddenStates
        hiddenStates = fc1(hiddenStates)
        hiddenStates = activationFn(hiddenStates)
        hiddenStates = fc2(hiddenStates)
        return hiddenStates
    }
}
