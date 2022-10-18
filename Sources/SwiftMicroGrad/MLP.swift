import Foundation

public class MLP {
    var sz: [Int]
    var layers: [Layer]
    
    init(_ nin: Int, _ nouts: [Int]) {
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
        loops: Int = 1000, stepForGradDescent: Double = 0.1, lossThreshold: Double = 0.0001, verbose: Bool = false) 
    {
        var finalLoss: Double = 1000.0
        var finalLoopNo = loops

        for i in 0...loops - 1 {
            if verbose { print("\n---------------Loop no \(i+1)") }

            //Forward Pass to estimate loss
            let ydred = xs.map { self.feed($0) }
            let losses = zip(ydred,ys).map() { ($0[0] - $1)**2 }
            var loss = losses[0]
            for i in 1..<losses.count {
                loss = loss + losses[i]
            }
            if verbose { print("Loss Before Gradient Descent \(loss.data)") }

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
        
            //Forward Pass to re-estimate loss
            let ydred2 = xs.map { self.feed($0) }
            let losses2 = zip(ydred2,ys).map() { ($0[0] - $1)**2 }
            var loss2 = losses2[0]
            for i in 1..<losses2.count {
                loss2 = loss2 + losses2[i]
            }
            if verbose { print(" Loss After Gradient Descent \(loss2.data)") }
            if verbose { print("----------------End loop no \(i+1)") }

            //exit when lossThreshold is achieved
            if loss2.data <= lossThreshold {
                finalLoopNo = i+1
                finalLoss = loss2.data
                break
            }

            //record final loss
            if (i+1) == loops {
                finalLoss = loss2.data
            }
        }

        let ydred = xs.map { self.feed($0) }

        if verbose {
            print("\nFinal prediction after \(finalLoopNo) loops with loss \(finalLoss)")
            print("Target \(ys)")
            print("Guess  \(ydred)")
        }

    }
}
