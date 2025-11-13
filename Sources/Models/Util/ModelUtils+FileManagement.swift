import Foundation

extension ModelUtils {
    static func isRemote(_ url: String) -> Bool {
        guard let urlComponents = URLComponents(string: url),
              let scheme = urlComponents.scheme?.lowercased()
        else {
            return false
        }

        let remoteSchemes = ["http", "https"]
        return remoteSchemes.contains(scheme) && urlComponents.host != nil
    }
    
    enum LocalURLStatus {
        case file
        case directory
        case notLocal
    }
    
    static func isLocal(_ url: String, _ subFolder: String? = nil) -> LocalURLStatus {
        let fileManager = FileManager.default
        var isDirectory: ObjCBool = false
        var isLocalEntity: Bool = false
        
        if let subfolder = subFolder, !subfolder.isEmpty {
            let urlNSString = url as NSString
            let directory = urlNSString.deletingLastPathComponent
            let filename = urlNSString.lastPathComponent
                                
            // Insert subfolder between directory and filename: /something/subfolder/file.txt
            let filePath = (directory as NSString).appendingPathComponent(subfolder).appending("/\(filename)")
            isLocalEntity = fileManager.fileExists(atPath: filePath, isDirectory: &isDirectory)
        } else {
            isLocalEntity = fileManager.fileExists(atPath: url, isDirectory: &isDirectory)
        }

        if isLocalEntity {
            return isDirectory.boolValue ? .directory : .file
        } else {
            return .notLocal
        }
    }
    
    /// - Note: This is a deprecated way to get the config file, Hub API should be used instead
    static func downloadUrl(_ url: String) async throws -> String {
        guard let downloadURL = URL(string: url) else {
            ModelUtils.log("Error: Invalid URL string: \(url)")
            throw AutoModelError.invalidConfig
        }

        // Create a temporary file URL with random filename
        let temporaryDirectoryURL = FileManager.default.temporaryDirectory
        let randomFileName = UUID().uuidString + ".json"
        let temporaryFileURL = temporaryDirectoryURL.appendingPathComponent(randomFileName)

        ModelUtils.log("Downloading from: \(url)")
        ModelUtils.log("Saving to temporary location: \(temporaryFileURL.path)")

        let (localURL, response) = try await URLSession.shared.download(from: downloadURL)

        // Validate HTTP response
        guard let httpResponse = response as? HTTPURLResponse,
              (200 ... 299).contains(httpResponse.statusCode)
        else {
            ModelUtils.log("Error: Invalid response type")
            throw AutoModelError.invalidConfig
        }

        do {
            // Move downloaded file to temporary location
            try FileManager.default.moveItem(at: localURL, to: temporaryFileURL)
            ModelUtils.log("Successfully downloaded file to: \(temporaryFileURL.path)")
            return temporaryFileURL.path
        } catch {
            ModelUtils.log("Error moving file: \(error.localizedDescription)")
            throw error
        }
    }
}
