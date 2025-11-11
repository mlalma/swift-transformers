import Foundation

extension PreTrainedModel {
    
    private func loadStateDict() {
        
    }
    
    private func loadShardFile(
        shardFile: String?,
        stateDict: [String: Any]?,
        weightsOnly: Bool
    ) {
        
    }
    
    internal func loadPreTrainedModel(
        model: PreTrainedModel,
        stateDict: [String: Any]?,
        checkpointFiles: [String]?,
        pretrainedModelNameOrPath: String?,
        ignoreMismatchedSizes: Bool = false,
        sharedMetadata: ShardedIndexFile? = nil,
        weightsOnly: Bool = true)
    async throws {
        // TO_DO: Key-name remapping
        // TO_DO: Quantization handling
        
        if let checkpointFiles {
            for shardFile in checkpointFiles {
                loadShardFile(shardFile: shardFile, stateDict: stateDict, weightsOnly: weightsOnly)
            }
        } else if let stateDict {
            loadShardFile(shardFile: nil, stateDict: stateDict, weightsOnly: weightsOnly)
        } else {
            throw AutoModelError.noModelDataToLoad
        }
    }
}
