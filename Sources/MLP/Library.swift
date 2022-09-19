import Foundation

public struct Value: CustomStringConvertible {
    var data, grad: Double
    var prev: [Value]
    var op: String
    var label: String
    
    init(_ data: Double, _ children:[Value] = [], _ op:String = "", label:String = "") {
        self.data = data
        self.grad = 0.0
        self.prev = children
        self.op = op
        self.label = label
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
    
    static func +(lhs: Value, rhs: Value) -> Value {
        return Value(lhs.data + rhs.data, [lhs, rhs],"+")
    }
    
    static func *(lhs: Value, rhs: Value) -> Value {
        return Value(lhs.data * rhs.data, [lhs, rhs],"*")
    }

    public func tanh() -> Value {
        let x: Double = self.data
        let t: Double = ( exp(2.0*x) - 1.0 ) / ( exp(2.0*x) + 1.0)
        return Value(t , [self], "α") //α stands for tanh
    }
    
    public var description: String {
        return "Value(data=\(self.data))"
    }
}