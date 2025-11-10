import Foundation

enum ModelUtils {
    static let version = "5.0.0-dev.0"
}

extension ModelUtils {
    static func isRemoteURL(_ url: String) -> Bool {
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
    
    static func isLocalURL(_ url: String, _ subFolder: String? = nil) -> LocalURLStatus {
        // Check if pretrainedModelNameOrPath is a local directory
        var isDirectory: ObjCBool = false
        let fileManager = FileManager.default
        var isLocal = fileManager.fileExists(atPath: url, isDirectory: &isDirectory) && isDirectory.boolValue

        // Check if it's a direct file path
        if let subfolder = subFolder, !subfolder.isEmpty {
            // Split the url into directory and filename
            let urlNSString = url as NSString
            let directory = urlNSString.deletingLastPathComponent
            let filename = urlNSString.lastPathComponent
                                
            // Insert subfolder between directory and filename: /opt/something/newfolder/file.txt
            let filePath = (directory as NSString).appendingPathComponent(subfolder).appending("/\(filename)")
            let isFile = fileManager.fileExists(atPath: filePath) && !isDirectory.boolValue
            
            if isFile { return .file }
        } else {
            // No subfolder, check url directly as file
            let isFile = fileManager.fileExists(atPath: url) && !isDirectory.boolValue
            if isFile { return .file }
        }

        if isLocal { return .directory }
        return .notLocal
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
