import Foundation
import Hub

extension PreTrainedModel {
    private static func cachedFile(_ pathOrRepoId: String, fileNames: [String], downloadArguments: DownloadArguments?) async throws -> [String]? {
        let subFolder = downloadArguments?.subFolder != nil ? downloadArguments!.subFolder : ""
        let revision = downloadArguments?.revision ?? "main"
        var cacheDir = downloadArguments?.cacheDir
    
        let fullFileNames = subFolder.isEmpty ? fileNames : fileNames.map { subFolder.hasSuffix("/") ? "\(subFolder)\($0)" : "\(subFolder)/\($0)"}
        
        var existingFiles: [String] = []
        for fileName in fullFileNames {
            if ModelUtils.isLocal(pathOrRepoId) == .directory {
                let resolvedFile = (pathOrRepoId + "/" + fileName).replacingOccurrences(of: "//", with: "/")
                if ModelUtils.isLocal(resolvedFile) != .file {
                    ModelUtils.log("Missing entry \(resolvedFile)")
                } else {
                    existingFiles.append(resolvedFile)
                }
            }
        }
        
        if ModelUtils.isLocal(pathOrRepoId) == .directory, existingFiles.count > 0 {
            return existingFiles
        }
        
        let downloadedRepoDir = try await HubApi.shared.snapshot(from: pathOrRepoId, revision: revision, matching: fullFileNames)
        
        existingFiles = []
        for fileName in fullFileNames {
            let fullFileNamePath = downloadedRepoDir.appendingPathComponent(fileName).path()
            if ModelUtils.isLocal(fullFileNamePath) == .file {
                existingFiles.append(fullFileNamePath)
            }
        }
        return existingFiles
    }
    
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
    ) async throws -> ([URL]?, [String: Any]?) {
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
        var fileName: String?
        var isSharded: Bool?
        
        guard ggutFile == nil else {
            ModelUtils.log("GGUF files are not yet supported")
            return (nil, nil)
        }

        let pretrainedModelNameOrPathType = ModelUtils.isLocal(pretrainedModelNameOrPath)
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
                    let filePath = URL(filePath: pretrainedModelNameOrPath)
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
                    let filePath = URL(filePath: pretrainedModelNameOrPath)
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
        } else if ModelUtils.isRemote(pretrainedModelNameOrPath) {
            fileName = pretrainedModelNameOrPath
            resolvedArchiveFile = URL(fileURLWithPath: try await ModelUtils.downloadUrl(pretrainedModelNameOrPath))
        } else {
            if let transformersExplicitFilename {
                fileName = transformersExplicitFilename
                isSharded = transformersExplicitFilename.hasPrefix(".safetensors.index.json")
            } else if let useSafetensors, useSafetensors {
                fileName = addVariant(weightsName: "model.safetensors", variant: variant)
            } else {
                fileName = addVariant(weightsName: "pytorch_model.bin", variant: variant)
            }
            
            if let fileName {
                if let resolvedFileStr = try await cachedFile(
                    pretrainedModelNameOrPath,
                    fileNames: [fileName],
                    downloadArguments: downloadArguments)?.first {
                    resolvedArchiveFile = URL(fileURLWithPath: resolvedFileStr)
                }
            }
        }
        
        var checkpointFiles: [URL]? = []
        if let isSharded, isSharded {
            // TO_DO: Download checkpoint files here
        } else {
            if let resolvedArchiveFile {
                checkpointFiles = [resolvedArchiveFile]
            }
        }
        
        return (checkpointFiles, nil)
    }

    static func fromPretrained(
        _ pretrainedModelNameOrPath: String,
        config: PreTrainedConfig,
        useSafetensors: Bool?,
        modelArguments: [String: Any]
    ) async throws -> PreTrainedModel? {
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

        if stateDict != nil {
            ModelUtils.log("State dict containing model weights is explicitly provided")
        } else {
            _ = try await getResolvedCheckpointFiles(
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
