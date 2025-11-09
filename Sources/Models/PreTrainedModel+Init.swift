import Foundation

extension PreTrainedModel {
    private static func getResolvedCheckpointFiles(
        _ pretrainedModelNameOrPath: String,
        variant: String? = nil,
        ggutFile: String? = nil,
        useSafetensors: Bool?,
        downloadArguments: DownloadArguments?,
        userAgent: [String: Any],
        transformersExplicitFilename: String? = nil
    ) {
        // TO_DO: Load model files here
    }

    static func fromPretrained(
        _ pretrainedModelNameOrPath: String,
        config: PreTrainedConfig,
        useSafetensors: Bool?,
        modelArguments: [String: Any]
    ) -> PreTrainedModel? {
        let stateDict = modelArguments["state_dict"] as? [String: Any]
        let ggufFile = modelArguments["gguf_file"] as? String
        let variant = modelArguments["variant"] as? String
        let downloadArgs = DownloadArguments(
            cacheDir: modelArguments["cache_dir"] as? String,
            forceDownload: modelArguments["force_download"] as? Bool ?? false,
            proxies: modelArguments["proxies"] as? [String: String],
            localFilesOnly: modelArguments["local_files_only"] as? Bool ?? false,
            token: modelArguments["token"] as? String,
            revision: modelArguments["revision"] as? String,
            subFolder: modelArguments["subfolder"] as? String ?? "",
            commitHash: nil)
        
        if ggufFile != nil {
            ModelUtils.log("GGUF files are not yet supported")
            return nil
        }
        
        let userAgent: [String: Any] = [
            "file_type": "model",
            "framework": "pytorch",
            "from_auto_class": true
        ]

        if let stateDict {
            ModelUtils.log("State dict containing model weights is explicitly provided")
        } else {
            getResolvedCheckpointFiles(
                pretrainedModelNameOrPath,
                variant: variant,
                ggutFile: nil,
                useSafetensors: useSafetensors,
                downloadArguments: downloadArgs,
                userAgent: userAgent,
                transformersExplicitFilename: config.transformersWeights
            )
        }
        
        return nil
    }
}
