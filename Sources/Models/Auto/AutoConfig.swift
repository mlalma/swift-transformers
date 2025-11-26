import Foundation
import Hub

class AutoConfig {
    private init() {}

    static func from(pretrained pretrainedModelNameOrPath: String, modelArguments: [String: Any] = [:]) async -> PreTrainedConfig? {
        if let config = await PreTrainedConfig.getConfigDict(pretrainedModelNameOrPath, modelArguments: modelArguments),
           let modelName: String = config[PreTrainedConfig.ConfigKeys.modelType, String.self],
           let configCreator = ModelMapping.configNames[modelName]
        {
            return configCreator(config)
        }

        return nil
    }
}
