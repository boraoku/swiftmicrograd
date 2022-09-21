import Foundation

public class Neuron {
    var w: [Value]  //weights
    var b: Value    //bias
    
    init(_ nin: Int) {
        self.w = (0..<nin).map { _ in Value(Double.random(in: -1.0...1.0)) }
        self.b = Value(Double.random(in: -1.0...1.0))
    }

    public func feed(_ x:[Double]) -> Value {
        let xValue: [Value] = x.map { Value($0) }
        return self.feed(xValue)
    }

    public func feed(_ x:[Value]) -> Value {
        var sumproduct = Value(0.0)
        for (wi, xi) in zip(self.w, x) {
            sumproduct = sumproduct + wi * xi
        }

        let act = sumproduct + self.b
        let out = act.tanh()
        return out
    }

    public func parameters() -> [Value] {
        return self.w + [self.b]
    }
}

public class Layer {
    var neurons: [Neuron]
    
    init(_ nin: Int, _ nout: Int) {
        self.neurons = (0..<nout).map { _ in Neuron(nin) }
    }

    public func feed(_ x:[Double]) -> [Value] {
        let xValue: [Value] = x.map { Value($0) }
        return self.feed(xValue)
    }

    public func feed(_ x:[Value]) -> [Value] {
        let outs = self.neurons.map { $0.feed(x) }
        return outs
    }

    public func parameters() -> [Value] {
        var params: [Value] = []
        for neuron in self.neurons {
            let ps = neuron.parameters()
            params += ps
        }
        return params
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
