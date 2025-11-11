import Foundation
import Hub

extension PreTrainedModel {
    struct ShardedIndexFile: Codable {
        struct Metadata: Codable {
            let totalParameters: Int?
            let totalSize: Int?
        }
        let metadata: Metadata?
        let weightMap: [String: String]?
    }
    
    private static func shardFiles(
        _ preTrainedModelNameOrPath: String,
        indexFileName: String,
        downloadArguments: DownloadArguments?)
    async throws -> ([String]?, ShardedIndexFile?) {
        if ModelUtils.isLocal(indexFileName) != .file {
            ModelUtils.log("Could not find index file for sharded files \(indexFileName)")
            return (nil, nil)
        }
        
        let indexFileData = try Data(contentsOf: URL(filePath: indexFileName))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        guard let shardedIndexFile = try? decoder.decode(ShardedIndexFile.self, from: indexFileData),
              let shardedFilesList = shardedIndexFile.weightMap?.values else {
            ModelUtils.log("Could not parse JSON from the index file \(indexFileName)")
            return (nil, nil)
        }
        
        let fileList = Set(shardedFilesList).sorted()
        guard !fileList.isEmpty else {
            ModelUtils.log("Could not parse sharded files from the index file \(shardedIndexFile)")
            return (nil, nil)
        }
        
        var shardedFileNames: [String]? = []
        
        if ModelUtils.isLocal(preTrainedModelNameOrPath) == .directory {
            var localDirectory = preTrainedModelNameOrPath
            let subFolder = downloadArguments?.subFolder != nil ? downloadArguments!.subFolder : ""

            if !subFolder.isEmpty {
                localDirectory = (localDirectory + "/" + subFolder + "/").replacingOccurrences(of: "//", with: "/")
            }
                
            for file in fileList {
                if ModelUtils.isLocal(localDirectory + file) == .file {
                    shardedFileNames?.append(file)
                } else {
                    ModelUtils.log("Could not find sharded weights file on local directory: \(localDirectory + file)")
                }
            }
        } else {
            shardedFileNames = try await cachedFiles(preTrainedModelNameOrPath, fileNames: Array(fileList), downloadArguments: downloadArguments)
        }
        
        return (shardedFileNames, shardedIndexFile)
    }
    
    private static func cachedFiles(_ pathOrRepoId: String, fileNames: [String], downloadArguments: DownloadArguments?) async throws -> [String]? {
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
    ) async throws -> ([URL]?, ShardedIndexFile?) {
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
                if let resolvedFileStr = try await cachedFiles(
                    pretrainedModelNameOrPath,
                    fileNames: [fileName],
                    downloadArguments: downloadArguments)?.first {
                    resolvedArchiveFile = URL(fileURLWithPath: resolvedFileStr)
                }
            }
        }
        
        var checkpointFiles: [URL]? = []
        var shardedMetadata: ShardedIndexFile?
        
        if let isSharded, isSharded, let resolvedArchiveFile {
            var shardedFiles: [String]?
            (shardedFiles, shardedMetadata) = try await shardFiles(pretrainedModelNameOrPath, indexFileName: resolvedArchiveFile.path(), downloadArguments: downloadArguments)
            if let shardedFiles {
                shardedFiles.forEach { checkpointFiles?.append(URL(filePath: $0)) }
            }
        } else {
            if let resolvedArchiveFile {
                checkpointFiles = [resolvedArchiveFile]
            }
        }
        
        return (checkpointFiles, shardedMetadata)
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
