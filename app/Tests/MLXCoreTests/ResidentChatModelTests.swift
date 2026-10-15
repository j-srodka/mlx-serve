import XCTest
@testable import MLXCore

final class ResidentChatModelTests: XCTestCase {
    func testResidentChatModelIdsSkipsMediaAndUnloaded() {
        let chat = ModelInfo(name: "gemma", quantBits: 4, layers: 48, hiddenSize: 3840,
                             vocabSize: 1, contextLength: 8192, modelMaxTokens: 8192,
                             capabilities: ["chat"], loaded: true)
        let video = ModelInfo(name: "h3", quantBits: 4, layers: 0, hiddenSize: 0,
                              vocabSize: 0, contextLength: 0, modelMaxTokens: 0,
                              capabilities: ["video"], loaded: true)
        let idle = ModelInfo(name: "qwen", quantBits: 4, layers: 40, hiddenSize: 2048,
                             vocabSize: 1, contextLength: 8192, modelMaxTokens: 8192,
                             capabilities: ["chat"], loaded: false)
        XCTAssertEqual(ServerManager.residentChatModelIds(from: [chat, video, idle]), ["gemma"])
    }
}
