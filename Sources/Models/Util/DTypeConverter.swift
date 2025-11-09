import Foundation
import MLX

extension ModelUtils {
    /// Maps PyTorch storage class names to MLX data types.
    static func dtype(_ dtypeStr: String?) -> DType? {
        guard let dtypeStr else { return nil }

        switch dtypeStr {
        case "DoubleStorage":
            return .float64
        case "FloatStorage":
            return .float32
        case "HalfStorage":
            return .float16
        case "LongStorage":
            return .int64
        case "IntStorage":
            return .int32
        case "ShortStorage":
            return .int16
        case "CharStorage":
            return .int8
        case "ByteStorage":
            return .uint8
        case "BoolStorage":
            return .bool
        case "BFloat16Storage":
            return .bfloat16
        case "ComplexDoubleStorage":
            return nil
        case "CompleteFloatStorage":
            return .complex64
        case "QUInt8Storage":
            return .uint8
        case "QInt8Storage":
            return .int8
        case "QInt32Storage":
            return .int32
        case "QUInt4x2Storage":
            return .uint8
        case "QUInt2x4Storage":
            return .uint8
        default:
            return nil
        }
    }
}
