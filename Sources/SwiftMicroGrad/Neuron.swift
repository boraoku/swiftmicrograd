import Foundation

public class Neuron {
    var w: [Value]  //weights
    var b: Value    //bias
    
    init(_ nin: Int) {
        self.w = (0..<nin).map { _ in Value(Double.random(in: -1.0...1.0)) }
        self.b = Value(Double.random(in: -1.0...1.0))
    }

    public func feed(_ x:[Double]) -> Value {
        var sumproduct = 0.0
        for (wi, xi) in zip(self.w, x) {
            sumproduct += wi.data * xi
        }

        let act = Value(sumproduct) + self.b
        let out = act.tanh()
        //print("processed neuron")
        return out
    }
}

public class Layer {
    var neurons: [Neuron]
    
    init(_ nin: Int, _ nout: Int) {
        self.neurons = (0..<nout).map { _ in Neuron(nin) }
    }

    public func feed(_ x:[Double]) -> [Value] {
        //print("processed layer")
        let outs = self.neurons.map { $0.feed(x) }
        return outs
    }
}

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
        var out: [Value] = []

        for layer in self.layers {
            out = layer.feed(x)
        }
        return out
    }
}
