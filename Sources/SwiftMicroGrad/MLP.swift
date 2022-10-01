import Foundation

public class MLP {
    var sz: [Int]
    var layers: [Layer?]?
    
    init(_ nin: Int, _ nouts: [Int]) {
        self.sz = [nin] + nouts
        self.layers = []
        for i in 0..<nouts.count {
            self.layers!.append(Layer(self.sz[i], self.sz[i+1]))
        }
    }
    
    deinit {
        layers?.removeAll()
        layers = nil
        //print("MLP dealloc")
    }
    
    public func feed(_ x:[Double]) -> [Value?]? {
        let xValue: [Value?] = x.map { Value($0) }
        return feed(xValue)
    }
    
    public func feed(_ x:[Value?]) -> [Value?]? {
        weak var self_ = self
        var outs: [Value?] = x
        for layer in self_!.layers! {
            outs = layer!.feed(outs)!
        }
        return outs
    }
    
    public func parameters() -> [Value?]? {
        weak var self_ = self
        var params: [Value?]? = []
        for layer in self_!.layers! {
            var ps:[Value?]? = layer!.parameters()!
            params = params! + ps!
            ps?.removeAll()
            ps = nil
        }
        return params
    }
    
    public func train(inputs xs:[[Double]], outputs ys:[Double],
                      loops: Int = 1000, stepForGradDescent: Double = 0.1, lossThreshold: Double = 0.0001, verbose: Bool = false)
    {
        autoreleasepool {
            
            weak var self_ = self
            
            var finalLoss: Double = 1000.0
            var finalLoopNo = loops
            
            for i in 0...loops - 1 {
                if verbose { print("\n---------------Loop no \(i+1)") }
                
                //Forward Pass to estimate loss
                var ydred:[[Value?]?]? = xs.map { self_?.feed($0) }
                var losses:[Value?]? = zip(ydred!,ys).map() { ($0![0]! - $1)!**2 }
                var loss: Value? = losses![0]
                autoreleasepool {
                    for i in 1..<losses!.count {
                        loss = loss! + losses![i]!
                    }                    
                }
                if verbose { print("Loss Before Gradient Descent \(loss!.data)") }
                
                //Zerograd - important!
                for p in self_!.parameters()! {
                    p!.grad = 0.0
                }
                
                //Backward Pass
                loss!.backward()
                
                //Gradient Descent
                for p in self_!.parameters()! {
                    p!.data += -stepForGradDescent * p!.grad
                    //FIXME: is the leak here???
                }
                
                //Forward Pass to re-estimate loss
                var ydred2:[[Value?]?]? = xs.map { self_?.feed($0) }
                var losses2:[Value?]? = zip(ydred2!,ys).map() { ($0![0]! - $1)!**2 }
                var loss2:Value? = losses2![0]
                for i in 1..<losses2!.count {
                    loss2 = loss2! + losses2![i]!
                }
                if verbose { print(" Loss After Gradient Descent \(loss2!.data)") }
                if verbose { print("----------------End loop no \(i+1)") }
                
                //exit when lossThreshold is achieved
                if loss2!.data <= lossThreshold {
                    finalLoopNo = i+1
                    finalLoss = loss2!.data
                    break
                }
                
                //record final loss
                if (i+1) == loops {
                    finalLoss = loss2!.data
                }
                
                //deinits
                losses!.removeAll()
                losses = nil
                ydred!.removeAll()
                ydred = nil
                
                losses2!.removeAll()
                losses2 = nil
                ydred2!.removeAll()
                ydred2 = nil
                
                loss=nil
                loss2=nil
            }
            
            var ydred:[[Value?]?]? = xs.map { self_?.feed($0) }
            
            if verbose {
                print("\nFinal prediction after \(finalLoopNo) loops with loss \(finalLoss)")
                print("Target \(ys)")
                print("Guess  \(ydred!)")
            }
            
            ydred!.removeAll()
            ydred = nil
            
            //just trying dealloc a simple value
            //var bora: Value? = Value(1.0)
            //bora = nil
        }
    }
}
