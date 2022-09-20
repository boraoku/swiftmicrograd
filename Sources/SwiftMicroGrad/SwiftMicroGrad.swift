import Foundation

private func lambda() {
    return
}

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

    static func +(self: Value, other: Value) -> Value {
    
        let out = Value(self.data + other.data, [self, other],"+")

        func _backward() {
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        }

        out._backward = _backward
        return out
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

    public func tanh() -> Value {
        let x: Double = self.data
        let t: Double = ( exp(2.0*x) - 1.0 ) / ( exp(2.0*x) + 1.0)

        let out = Value(t , [self], "α") //α stands for tanh
        
        func _backward() {
            self.grad += (1 - pow(t,2.0)) * out.grad
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
        return "Value(data=\(self.data))"
    }
}