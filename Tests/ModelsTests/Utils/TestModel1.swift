import Foundation
import MLX
@testable import Models


final class TestModel1: PreTrainedModel {
    var stateDict: [String: MLXArray] = [:]
    
    override func loadWeightsToModel(_ weights: [String: MLXArray]) {
        stateDict = weights
    }
}
