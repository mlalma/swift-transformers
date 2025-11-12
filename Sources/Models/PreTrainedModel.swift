import Foundation
import MLX

public class PreTrainedModel {
    /// Hook to add instantiators to .pt file reader for resolving classes during the model loading.
    /// Should be overridden by derived model class in case it needs to add some specific instantiators to be able to read .pt / .bin files
    func addInstantiators() {
    }
    
    /// Entry point to add intialized weights to the model.
    /// Should be overridden by the derived model class.
    func loadWeightsToModel(_ weights: [String: MLXArray]) {
        ModelUtils.log("Should be overridden by the derived model class")
    }
}
