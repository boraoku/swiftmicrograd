import XCTest
@testable import SwiftMicroGrad

final class SwiftMicroGradTests: XCTestCase {
    func testMLPParameterCount() throws {
        let o = MLP(3, [4, 4, 1])
        XCTAssertEqual(o.parameters().count, 41)
    }
    
    func testCurveFitting() throws {
        
        let xs = [1.1, 1.5, 3.0, 6.0]         //inputs
        let ys = [0.25, 0.40, 0.75, 1.0]      //outputs
        
        let targets = [1.0, 2.0, 4.0, 5.0, 8.0]
        
        print("\nTraining...\n")
        let numberOfRuns = 5
        var results: [[Double]] = []

        for _ in 0 ..< targets.count {
            let innerArray = [Double](repeating: 0.0, count: numberOfRuns)
            results.append(innerArray)
        }
        let queue = OperationQueue()
        
        for i in 0..<numberOfRuns {
            queue.addOperation {
                print("Running MLP \(i+1)/\(numberOfRuns)")
                let n:MLP = MLP(1, [ys.count, ys.count, 1])
                n.train(inputs: xs, outputs: ys, loops:10000, stepForGradDescent: 0.05, lossThreshold: 10e-5, verbose: true, concurrencyCount: i+1)
                var j = 0
                for target in targets {
                    results[j][i] = n.feed([target])[0].data
                    j += 1
                }
            }
        }

        queue.waitUntilAllOperationsAreFinished()

        var averageResults: [Double] = []
        for i in 0..<targets.count {
            averageResults.append(results[i].reduce(0.0, +) / Double(numberOfRuns))
            
            let resultsString = results[i].map { String(format:"%.4f", $0)  }.joined(separator: ", ")
            print("\nGuesses for Load levels at \(String(format:"%.4f", targets[i]))mm Creep Rate: " + resultsString)
            print("                               In Average: " + String(format:"%.4f", averageResults[i]))
        }
        
        XCTAssertEqual(averageResults[1]*100, 53.0, accuracy: 1.0) //at 2mm, a good result shall be within ±1 of 53
    }
}
