import Foundation
import MLX
import MLXNN
import PTReaderSwift

extension PreTrainedModel {
    private func loadStateDict(
        checkpointFileName: URL,
        weightsOnly: Bool = true
    ) async throws -> [String: MLXArray]? {
        if checkpointFileName.path().hasSuffix(".safetensors") {
            let weights = try MLX.loadArrays(url: checkpointFileName)
            return weights
        } else {
            addInstantiators()
            
            let outputVal = try await Task { @PTReaderActor in
              let file = try PTFile(fileName: checkpointFileName)
              return file.parseData()
            }.value
            
            var weights: [String: MLXArray] = [:]
            if let outputVal, let outputDict = outputVal.dict {
                for (key, value) in outputDict {
                    if let key = key as? String,
                       let potentialArray = value as? (object: Any, objectType: String),
                       potentialArray.objectType == "Tensor",
                       let mlxArray = potentialArray.object as? MLXArray {
                        weights[key] = mlxArray
                    }
                }
            }
                        
            guard weights.keys.count > 0 else { return nil }
            return weights
        }
    }
    
    private func loadShardFile(
        shardFile: URL?,
        stateDict: [String: MLXArray]?,
        weightsOnly: Bool
    ) async throws {
        if let shardFile {
            guard let loadedDict = try await loadStateDict(checkpointFileName: shardFile, weightsOnly: weightsOnly) else {
                ModelUtils.log("Could not load state dictionary of the model")
                return
            }
            try loadWeightsToModel(loadedDict)
        } else if let stateDict {
            try loadWeightsToModel(stateDict)
        }
    }
    
    internal func loadPreTrainedModel(
        stateDict: [String: MLXArray]?,
        checkpointFiles: [URL]?,
        pretrainedModelNameOrPath: String?,
        ignoreMismatchedSizes: Bool = false,
        sharedMetadata: ShardedIndexFile? = nil,
        weightsOnly: Bool = true)
    async throws {
        // TO_DO: Key-name remapping
        // TO_DO: Quantization handling
        
        if let checkpointFiles {
            for shardFile in checkpointFiles {
                try await loadShardFile(shardFile: shardFile, stateDict: stateDict, weightsOnly: weightsOnly)
            }
        } else if let stateDict {
            try await loadShardFile(shardFile: nil, stateDict: stateDict, weightsOnly: weightsOnly)
        } else {
            throw AutoModelError.noModelDataToLoad
        }
    }
}
