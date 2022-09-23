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
}
