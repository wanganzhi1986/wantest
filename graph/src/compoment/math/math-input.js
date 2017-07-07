/**
 * Created by wangwei on 17/6/26.
 */
import React, { Component } from 'react';
import TexButtons from './tex-button'
import ReactDOM from 'react-dom';
import _ from 'underscore';
import './math.css';

var MathQuill = window.MathQuill;

//实现原理: 输入框获得焦点后弹出输入公式的面板，此时有两个事件：1.获得焦点事件 2.还要跟踪鼠标的移动
//因为当点击公式面板时，对于输入框而言，意味着失去了焦点，此时公式面板不能消失，因此通过函数handleMouseDown
//处理这种情况

class MathInput extends Component{
    constructor(props){
        super(props);
        this.state = {
            focused:false
        }
    }

    handleMouseDown(event) {
        var focused = ReactDOM.findDOMNode(this).contains(event.target);
        console.log(focused)
        this.mouseDown = focused;
        if (!focused) {
            this.setState({ focused: false });
    }
}

    handleMouseUp() {
        // this mouse click started in the buttons div so we should focus the
        // input
        if (this.mouseDown) {
            this.focus();
        }
        this.mouseDown = false;
}

    insert(value) {
        // console.log('value is:', value);
        var input = this.mathField();
        if (_(value).isFunction()) {
            value(input);
        } else if (value[0] === '\\') {
            input.cmd(value).focus();
        } else {
            input.write(value).focus();
        }
        input.focus();
    }

    mathField(options) {
        var MQ = MathQuill.getInterface(2);

        // MathQuill.MathField takes a DOM node, MathQuill-ifies it if it's
        // seeing that node for the first time, then returns the associated
        // MathQuill object for that node. It is stable - will always return
        // the same object when called on the same DOM node.
        return MQ.MathField(this.mathinput, options);
    }

    componentWillUnmount() {
        window.removeEventListener("mousedown",this.handleMouseDown.bind(this));
        window.removeEventListener("mouseup", this.handleMouseUp.bind(this));
    }

    componentDidMount() {
        window.addEventListener("mousedown", this.handleMouseDown.bind(this));
        window.addEventListener("mouseup", this.handleMouseUp.bind(this));

        var initialized = false;

    // Initialize MathQuill.MathField instance
        this.mathField({
        // LaTeX commands that, when typed, are immediately replaced by the
        // appropriate symbol. This does not include ln, log, or any of the
        // trig functions; those are always interpreted as commands.
        autoCommands: "pi theta phi sqrt nthroot",

        // Pop the cursor out of super/subscripts on arithmetic operators
        // or (in)equalities.
        charsThatBreakOutOfSupSub: "+-*/=<>≠≤≥",

        // Prevent excessive super/subscripts or fractions from being created
        // without operands, e.g. when somebody holds down a key
        supSubsRequireOperand: true,

        spaceBehavesLikeTab: true,

        handlers: {
            edited: (mathField) => {
                // This handler is guaranteed to be called on change, but
                // unlike React it sometimes generates false positives.
                // One of these is on initialization (with an empty string
                // value), so we have to guard against that below.
                var value = mathField.latex();
                console.log("现在的value is:", value);

                // Provide a MathQuill-compatible way to generate the
                // not-equals sign without pasting unicode or typing TeX
                value = value.replace(/<>/g, "\\ne");

                // Use the specified symbol to represent multiplication
                // TODO(alex): Add an option to disallow variables, in
                // which case 'x' should get converted to '\\times'
                if (this.props.convertDotToTimes) {
                    value = value.replace(/\\cdot/g, "\\times");

                    // Preserve cursor position in the common case:
                    // typing '*' to insert a multiplication sign.
                    // We do this by modifying internal MathQuill state
                    // directly, instead of waiting for `.latex()` to be
                    // called in `componentDidUpdate()`.
                    var left = mathField.__controller.cursor[MathQuill.L];
                    if (left && left.ctrlSeq === '\\cdot ') {
                        mathField.__controller.backspace();
                        mathField.cmd('\\times');
                    }
                } else {
                    value = value.replace(/\\times/g, "\\cdot");
                }

                if (initialized && this.props.value !== value) {
                    this.props.onChange(value);
                }
            },
            upOutOf: (mathField) => {
                // This handler is called when the user presses the up
                // arrow key, but there is nowhere in the expression to go
                // up to (no numerator or exponent). For ease of use,
                // interpret this as an attempt to create an exponent.
                mathField.typedText("^");
            }
        }
    });

        // Ideally, we would be able to pass an initial value directly into
        // the constructor above
        this.mathField().latex(this.props.value);
        initialized = true;
    }

    componentDidUpdate() {
        if (!_.isEqual(this.mathField().latex(), this.props.value)) {
            this.mathField().latex(this.props.value);
        }
    }

    focus() {
        this.mathField().focus();
        this.setState({ focused: true });
    }

    blur() {
        this.mathField().blur();
        this.setState({ focused: false });
    }

    handleFocus(){
        this.setState({focused: true});
    }

    handleBlur(){
        //在这里使用this.mouseDowm进行判断，如果没有这个条件，不会触发onclick事件
        if (!this.mouseDown) {
            this.setState({ focused: false });
        }
    }


    render(){
        var buttons = null;
        if (this.state.focused) {
            buttons = <TexButtons
                className="math-input-buttons absolute"
                onInsert={(value) =>this.insert(value)} />;
        }
        //onBlur={()=>this.handleBlur()}与{()=>this.handleBlur}出现的结果是不同的
        return (
            <div style={{display: "inline-block"}}>
                <div style={{display: 'inline-block'}}>

                <span
                      ref={(input)=>{this.mathinput=input}}
                      onFocus={()=>this.handleFocus}
                      onBlur={()=>this.handleBlur} />
            </div>
            <div style={{position: "relative"}}>
                {buttons}
            </div>
        </div>
        )
    }
}

export default MathInput