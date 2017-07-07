/**
 * Created by wangwei on 17/6/26.
 */
import React, { Component } from 'react';
import ReactDOM from 'react-dom';
import PureRenderMixin from 'react-addons-pure-render-mixin';
import katexA11y from './katex-a11y';


let pendingScripts = [];
let pendingCallbacks = [];
let needsProcess = false;

let IsmathJaxLoaded = false;


const process = (script, callback) => {
    pendingScripts.push(script);
    pendingCallbacks.push(callback);
    if (!needsProcess) {
        needsProcess = true;
        setTimeout(doProcess, 0);
    }
};

const loadMathJax = (callback) => {
    if (typeof MathJax !== "undefined") {
        callback();
    } else if (!IsmathJaxLoaded) {
        mathJaxLoaded();
        callback();
    } else {
        throw new Error(
            "MathJax wasn't loaded before it was needed by <TeX/>");
    }
};

const mathJaxLoaded = () =>{
    MathJax.Hub.Config({
        styles:{".MathJax_Display": {"text-align": "left"}},
        showProcessingMessages: false,
        tex2jax: { inlineMath: [['$','$'],['\\(','\\)']] }
    });
    IsmathJaxLoaded = true;
};



const doProcess = () => {
    loadMathJax(() => {
        MathJax.Hub.Queue(function() {
            const oldElementScripts = MathJax.Hub.elementScripts;
            //
            MathJax.Hub.elementScripts = (element) => pendingScripts;

            try {
                return MathJax.Hub.Process(null, () => {
                    // Trigger all of the pending callbacks before clearing them
                    // out.
                    // console.log("length is:", pendingCallbacks.length);
                    // if (pendingCallbacks.length > 0){
                    //     for (const callback of pendingCallbacks) {
                    //         console.log(typeof callback);
                    //         callback();
                    //     }
                    //
                    // }
                    pendingScripts = [];
                    pendingCallbacks = [];
                    needsProcess = false;
                });
            } catch (e) {
                // IE8 requires `catch` in order to use `finally`
                throw e;
            } finally {
                MathJax.Hub.elementScripts = oldElementScripts;
            }
        });
    });
};

// Make content only visible to screen readers.
// Both collegeboard.org and Bootstrap 3 use this exact implementation.
const srOnly = {
    border: 0,
    clip: "rect(0,0,0,0)",
    height: "1px",
    margin: "-1px",
    overflow: "hidden",
    padding: 0,
    position: "absolute",
    width: "1px",
};


class Tex extends Component{
    constructor(props){
        super(props);
        this.state = {};

        //添加PureRenderMixin特质
        this.shouldComponentUpdate = PureRenderMixin.shouldComponentUpdate.bind(this);

    }

    // static defaultProps = {
    //     onRender: ()=>{}
    //
    // };

    //设置props的默认值
    //static defaultProps = {
    //  name: 'Mary'  //定义defaultprops的另一种方式
    //}

    //static propTypes = {
    //name: React.PropTypes.string
    //}

    componentDidMount(){
        this._root = ReactDOM.findDOMNode(this);

        if (this.refs.katex.childElementCount > 0) {
            // If we already rendered katex in the render function, we don't
            // need to render anything here.
            // this.props.onRender(this._root);
            return;
        }

        const text = this.props.children;

        this.setScriptText(text);
        process(this.script);
    }

    componentDidUpdate(prevProps, prevState){
        if (this.refs.katex.childElementCount > 0) {
            if (this.script) {
                // If we successfully rendered KaTeX, check if there's
                // lingering MathJax from the last render, and if so remove it.
                loadMathJax(() => {
                    const jax = MathJax.Hub.getJaxFor(this.script);
                    if (jax) {
                        jax.Remove();
                    }
                });
            }

            // this.props.onRender();
            return;
        }

        const newText = this.props.children;

        if (this.script) {
            loadMathJax(() => {
                MathJax.Hub.Queue(() => {
                    const jax = MathJax.Hub.getJaxFor(this.script);
                    if (jax) {
                        // return jax.Text(newText, this.props.onRender);
                        return jax.Text(newText);
                    } else {
                        this.setScriptText(newText);
                        // process(this.script, this.props.onRender);
                        process(this.script);
                    }
                });
            });
        } else {
            this.setScriptText(newText);
            // process(this.script, this.props.onRender);
            process(this.script);
        }

    }

    componentWillUnmount() {
    if (this.script) {
        loadMathJax(() => {
            const jax = MathJax.Hub.getJaxFor(this.script);
            if (jax) {
                jax.Remove();
            }
        });
    }
}
    //创建<script>对象
    setScriptText(text) {
    //判断<script>对象是否创建，如果没有，则创建此对象，否则只需要修改text值即可
    if (!this.script) {
        this.script = document.createElement("script");
        this.script.type = "math/tex";
        ReactDOM.findDOMNode(this.refs.mathjax).appendChild(this.script);
    }
    if ("text" in this.script) {
        // IE8, etc
        this.script.text = text;
    } else {
        this.script.textContent = text;
    }
    }

    render() {
        let katexHtml = null;
        try {
            katexHtml = {
                __html: katex.renderToString(this.props.children),
            };
        } catch (e) {
            /* jshint -W103 */
            if (e.__proto__ !== katex.ParseError.prototype) {
                /* jshint +W103 */
                throw e;
            }
        }

        let katexA11yHtml = null;
        if (katexHtml) {
            try {
                katexA11yHtml = {
                    __html: katexA11y.renderString(this.props.children),
                };
            } catch (e) {
                // Nothing
            }
        }

    return <span
        style={this.props.style}
        onClick={this.props.onClick}
        >
            <span ref="mathjax" />
            <span
                ref="katex"
                dangerouslySetInnerHTML={katexHtml}
                aria-hidden={!!katexHtml && !!katexA11yHtml}
            />
            <span
                dangerouslySetInnerHTML={katexA11yHtml}
                style={srOnly}
            />
        </span>;
    }
}

export default Tex

