import Foundation

public struct Value: CustomStringConvertible {
    var data/*, grad*/: Double
    var prev: [Value]
    var op: String
    var label: String
    
    init(_ data: Double, _ children:[Value] = [], _ op:String = "", label:String = "") {
        self.data = data
        self.prev = children
        self.op = op
        self.label = label
        /*self.grad = grad*/
    }

    public func drawDot(_ level:Int = 0) -> String {

        let spacing = String(repeating: " |", count: level)

        var drawResult = "\(spacing)_\(self.label) data \(self.data)|"

        if self.prev.count > 1 {

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
    
    public var description: String {
        return "Value(data=\(self.data))"
    }
}