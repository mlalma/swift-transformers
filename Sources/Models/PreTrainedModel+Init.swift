import Foundation

extension PreTrainedModel {
    private static func addVariant(weightsName: String, variant: String?) -> String {
        if let variant {
            var weightParts = weightsName.split(separator: ".").map(String.init)
            weightParts.insert(variant, at: weightParts.count - 1)
            return weightParts.joined(separator: ".")
        }
        return weightsName
    }
    
    private static func getResolvedCheckpointFiles(
        _ pretrainedModelNameOrPath: String,
        variant: String? = nil,
        ggutFile: String? = nil,
        useSafetensors: Bool?,
        downloadArguments: DownloadArguments?,
        userAgent: [String: Any],
        transformersExplicitFilename: String? = nil
    ) async throws -> ([String]?, [String: Any]?) {
        let subFolder = downloadArguments?.subFolder ?? ""
        
        if let transformersExplicitFilename {
            if transformersExplicitFilename.hasSuffix(".safetensors") || transformersExplicitFilename.hasSuffix(".safetensors.index.json") {
                ModelUtils.log("The transformers file in the config seems to be incorrect: it is neither a safetensors file " +
                               "(*.safetensors) nor a safetensors index file (*.safetensors.index.json): \(transformersExplicitFilename)")
            }
            return (nil, nil)
        }
        
        var archiveFile: URL?
        var resolvedArchiveFile: URL?
        var isSharded: Bool?
        
        guard ggutFile == nil else {
            ModelUtils.log("GGUF files are not yet supported")
            return (nil, nil)
        }

        let pretrainedModelNameOrPathType = ModelUtils.isLocalURL(pretrainedModelNameOrPath)
        if pretrainedModelNameOrPathType == .directory {
            if let transformersExplicitFilename {
                archiveFile = URL(filePath: pretrainedModelNameOrPath)
                    .appending(path: subFolder)
                    .appending(component: transformersExplicitFilename)
                isSharded = transformersExplicitFilename.hasSuffix(".safetensors.index.json")
            } else if let useSafetensors, useSafetensors {
                let fileManager = FileManager.default
                let weights = [
                    ("model.safetensors", false) ,
                    ("model.safetensors.index.json", true),
                ]
                
                for (fileName, weightsAreSharded) in weights {
                    var filePath = URL(filePath: pretrainedModelNameOrPath)
                        .appending(path: subFolder)
                        .appending(component: addVariant(weightsName: fileName, variant: variant))
                    if fileManager.fileExists(atPath: filePath.path()) {
                        archiveFile = filePath
                        isSharded = weightsAreSharded
                        break
                    }
                }
                
                if archiveFile == nil {
                    ModelUtils.log("Error - could not find a safetensor model file in the given path: \(pretrainedModelNameOrPath)")
                    return (nil, nil)
                }
            } else {
                let fileManager = FileManager.default
                let weights = [
                    ("pytorch_model.bin", false),
                    ("pytorch_model.bin.index.json", true)
                ]
                
                for (fileName, weightsAreSharded) in weights {
                    var filePath = URL(filePath: pretrainedModelNameOrPath)
                        .appending(path: subFolder)
                        .appending(component: addVariant(weightsName: fileName, variant: variant))
                    if fileManager.fileExists(atPath: filePath.path()) {
                        archiveFile = filePath
                        isSharded = weightsAreSharded
                        break
                    }
                }
                
                if archiveFile == nil {
                    ModelUtils.log("Error - could not find a pytorch model file in the given path: \(pretrainedModelNameOrPath)")
                    return (nil, nil)
                }
            }
        } else if pretrainedModelNameOrPathType == .file {
            archiveFile = URL(filePath: pretrainedModelNameOrPath)
        } else if ModelUtils.isRemoteURL(pretrainedModelNameOrPath) {
            resolvedArchiveFile = URL(fileURLWithPath: try await ModelUtils.downloadUrl(pretrainedModelNameOrPath))
        }
        
        // TO_DO: Load model files here
        return (nil, nil)
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
