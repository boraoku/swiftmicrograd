import Foundation

public class MLP {
    public var sz: [Int]
    public var layers: [Layer]
    
    public init(_ nin: Int, _ nouts: [Int]) {
        self.sz = [nin] + nouts
        self.layers = []
        for i in 0..<nouts.count {
            self.layers.append(Layer(self.sz[i], self.sz[i+1]))
        }
    }
    
    deinit {
        layers.removeAll()
        //print("MLP dealloc")
    }

    public func feed(_ x:[Double]) -> [Value] {
        let xValue: [Value] = x.map { Value($0) }
        return self.feed(xValue)
    }

    public func feed(_ x:[Value]) -> [Value] {
        var outs = x
        for layer in self.layers {
            outs = layer.feed(outs)
        }
        return outs
    }

    public func parameters() -> [Value] {
        var params: [Value] = []
        for layer in self.layers {
            let ps = layer.parameters()
            params += ps
        }
        return params
    }

    public func train(inputs xs:[[Double]], outputs ys:[Double], 
                      loops: Int = 1000, stepForGradDescent: Double = 0.1, lossThreshold: Double = 0.0001, verbose: Bool = false, concurrencyCount: Int = -1)
    {
        var finalLoss: Double = 1000.0
        var finalLoopNo = loops

        for i in 0...loops - 1 {
            
            //Forward Pass to estimate loss
            let ydred = xs.map { self.feed($0) }
            let losses = zip(ydred,ys).map() { ($0[0] - $1)**2 }
            var loss = losses[0]
            for i in 1..<losses.count {
                loss = loss + losses[i]
            }
            var concurrency = "---------------"
            if concurrencyCount > 0 { concurrency = "--Thread no \(concurrencyCount)--" }
            if verbose { print("\n\(concurrency) Loop no \(i+1) Loss \(loss.data)") }
            
            //exit when lossThreshold is achieved
            if loss.data <= lossThreshold {
                finalLoopNo = i+1
                finalLoss = loss.data
                break
            }

            //Zerograd - important!
            for p in self.parameters() {
                p.grad = 0.0
            }

            //Backward Pass
            loss.backward()

            //Gradient Descent
            for p in self.parameters() {
                p.data += -stepForGradDescent * p.grad
            }

            //record final loss
            if (i+1) == loops {
                finalLoss = loss.data
            }
        }

        let ydred = xs.map { self.feed($0) }

        if verbose {
            print("\nFinal prediction after \(finalLoopNo) loops with loss \(finalLoss)")
            print("Training data outputs \(ys)")
            print("Prediction outputs    \(ydred)")
        }

    }
}
