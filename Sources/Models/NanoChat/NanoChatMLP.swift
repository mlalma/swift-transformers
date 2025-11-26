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
    /// - Parameters:
    ///   - config: Configuration for the NanoChat model.
    ///   - layerIdx: Index of the layer in the model.
    ///   - weights: Dictionary of weight tensors.
    init(config: NanoChatConfig, layerIdx: Int, weights: [String: MLXArray]) throws {
        self.config = config
        
        // Initialize activation function based on config
        guard let activation = ActivationFunction(activationFunction: config.hiddenAct) else {
            throw AutoModelError.invalidConfig("Unknown activation function: \(config.hiddenAct)")
        }
        self.activationFn = activation
        
        // Build weight keys for this layer
        let fc1WeightKey = Constants.weightKey(layerIdx: layerIdx, fcName: "fc1")
        let fc2WeightKey = Constants.weightKey(layerIdx: layerIdx, fcName: "fc2")
        
        // Verify all required weights exist
        guard let fc1Weight = weights[fc1WeightKey] else {
            throw AutoModelError.invalidConfig("Missing weight: \(fc1WeightKey)")
        }
        guard let fc2Weight = weights[fc2WeightKey] else {
            throw AutoModelError.invalidConfig("Missing weight: \(fc2WeightKey)")
        }
        
        // Initialize linear layers without bias
        self.fc1 = Linear(weight: fc1Weight)
        self.fc2 = Linear(weight: fc2Weight)
        
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
    
    // MARK: - Constants
    
    struct Constants {
        static let layersPrefix = "model.layers"
        static let mlpPrefix = "mlp"
        static let weightSuffix = "weight"
        
        /// Generates the weight key for a specific layer and fully connected layer.
        /// - Parameters:
        ///   - layerIdx: The layer index.
        ///   - fcName: The fully connected layer name (e.g., "fc1", "fc2").
        /// - Returns: The full weight key string.
        static func weightKey(layerIdx: Int, fcName: String) -> String {
            return "\(layersPrefix).\(layerIdx).\(mlpPrefix).\(fcName).\(weightSuffix)"
        }
    }
}
