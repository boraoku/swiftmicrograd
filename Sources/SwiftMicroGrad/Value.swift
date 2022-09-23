import Foundation

private func lambda() {
    return
}

infix operator **

public class Value: CustomStringConvertible {
    var data: Double
    var grad: Double
    var prev: [Value]
    var op: String
    var label: String
    var _backward: () -> () = lambda
    var topoVisited: Bool = false

    init(_ data: Double, _ children:[Value] = [], _ op:String = "", label:String = "", _ grad: Double = 0.0 ) {
        self.data = data
        self.grad = grad
        self.prev = children
        self.op = op
        self.label = label
    }

    static func +(self: Value, other: Double) -> Value {
        return self + Value(other)
    }

    static func +(other: Double, self: Value) -> Value {
        return self + Value(other)
    }

    static func +(self: Value, other: Value) -> Value {
    
        let out = Value(self.data + other.data, [self, other],"+")

        func _backward() {
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        }

        out._backward = _backward
        return out
    }

    static func -(self: Value, other: Double) -> Value {
        return self + (-1) * Value(other)
    }

    static func -(other: Double, self: Value) -> Value {
        return self + (-1) * Value(other)
    }

    static func -(self: Value, other: Value) -> Value {
        return self + (-1) * other
    }

    static func *(self: Value, other: Double) -> Value {
        return self * Value(other)
    }

    static func *(other: Double, self: Value) -> Value {
        return self * Value(other)
    }

    static func *(self: Value, other: Value) -> Value {

        let out = Value(self.data * other.data, [self, other],"*")

        func _backward() {
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        }

        out._backward = _backward
        return out
    }

    static func **(self: Value, other: Double) -> Value {

        let out = Value(pow(self.data, other), [self], String(format:"** %.4f", other))

        func _backward() {
            self.grad += other * pow(self.data, (other - 1)) * out.grad
        }

        out._backward = _backward
        return out
    }

    static func /(self: Value, other: Value) -> Value {
        return self * (other**(-1.0))
    }

    public func tanh() -> Value {
        let x: Double = self.data
        let t: Double = ( Darwin.exp(2.0*x) - 1.0 ) / ( Darwin.exp(2.0*x) + 1.0)

        let out = Value(t , [self], "α") //α stands for tanh
        
        func _backward() {
            self.grad += (1 - pow(t,2.0)) * out.grad
        }

        out._backward = _backward
        return out
    }

    public func exp() -> Value {
        let x: Double = self.data

        let out = Value(Darwin.exp(x) , [self], "e") //e stands for exp
        
        func _backward() {
            self.grad += out.data * out.grad
        }

        out._backward = _backward
        return out   
    }

    public func backward() {
        var topo :[Value] = []

        func buildTopo(_ v:Value) {
            if !v.topoVisited {
                v.topoVisited = true
                for child in v.prev {
                    buildTopo(child)
                }
                topo.append(v)
            }
        }

        buildTopo(self)

        self.grad = 1.0
        for node in topo.reversed() {
            node._backward()
        }
    }
    
    public func drawDot(_ level:Int = 0) -> String {

        let spacing = String(repeating: " |", count: level)

        var drawResult = "\(spacing)_\(self.label)" + String(format:" data %.4f", self.data) + String(format:" grad %.4f", self.grad)

        if self.prev.count > 0 {

            var printOp = true
    
            for child in self.prev {
    
                drawResult += "\n\(child.drawDot(level+1))"
                
                if printOp {
                    drawResult += "\n\(spacing) \(self.op)"
                    printOp = false
                }
            }
        }

        return drawResult
    }

    public var description: String {
        return "Value(data=" + String(format:" data %.4f", self.data)
    }
}