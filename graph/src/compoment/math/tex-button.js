/**
 * Created by wangwei on 17/6/26.
 */
import React, { Component } from 'react';
import _ from "underscore";
import TeX from './tex';


let prettyBig = { fontSize: "150%" };
var slightlyBig = { fontSize: "120%" };
var symbStyle = { fontSize: "130%" };

var buttonSets = {
    "basic":[
        [<span key="plus">+</span>, "+"],
        [<span key="minus">-</span>, "-"],
        [<TeX key="times">\times</TeX>, "\\times"],
        [<TeX key="cdot">\cdot</TeX>, "\\cdot"],
        [<TeX key="frac" style={{fontSize: "50%"}}>{"\\frac{□}{□}"}</TeX>,
            input => {
                var contents = input.latex();
                input.typedText("/");
                if (input.latex() === contents) {
                    input.cmd("\\frac");
                }
            }
        ]

    ],
    "trig": [
        [<TeX key="sin">\sin</TeX>, "\\sin"],
        [<TeX key="cos">\cos</TeX>, "\\cos"],
        [<TeX key="tan">\tan</TeX>, "\\tan"],
        [<TeX key="theta" style={symbStyle}>\theta</TeX>, "\\theta"],
        [<TeX key="pi" style={symbStyle}>\phi</TeX>, "\\phi"]
    ],


    "logarithm": [
        [<TeX key="log">\log</TeX>, "\\log"],
        [<TeX key="ln">\ln</TeX>, "\\ln"],
        [<TeX key="log_b">\log_b</TeX>,
            input => {
                input.typedText("log_");
                input.keystroke("Right");
                input.typedText("(");
                input.keystroke("Left");
                input.keystroke("Left");
            }],
        [<TeX key="sqrt">{"\\sqrt{x}"}</TeX>, "\\sqrt"],
        [<TeX key="nthroot">{"\\sqrt[3]{x}"}</TeX>, input => {
                input.typedText("nthroot3");
                input.keystroke("Right");
            }],
        [<TeX key="pow" style={slightlyBig}>□^a</TeX>, input => {
                input.latex("x^2");
                // var contents = input.latex();
                // input.typedText("^");
                // // If the input hasn't changed (for example, if we're
                // // attempting to add an exponent on an empty input or an empty
                // // denominator), insert our own "a^b"
                // if (input.latex() === contents) {
                //     input.typedText("a^b");
                // }
            }
        ],

    ],

    "relation": [
        [<TeX key="eq">{"="}</TeX>, "="],
        [<TeX key="lt">\lt</TeX>, "\\lt"],
        [<TeX key="gt">\gt</TeX>, "\\gt"],
        [<TeX key="neq">\neq</TeX>, "\\neq"],
        [<TeX key="leq">\leq</TeX>, "\\leq"],
        [<TeX key="geq">\geq</TeX>, "\\geq"],
    ],

   "geometry": [
       [<TeX key="rightarrow">\rightarrow</TeX>, "\\rightarrow"],
       [<TeX key="leftrightarrow">\leftrightarrow</TeX>, "\\leftrightarrow"],
       [<TeX key="overarc">{"\\overarc{AB}"}</TeX>, "\\overarc"],
       [<TeX key="overline">{"\\overline{AB}"}</TeX>, "\\overline"],
       [<TeX key="parallel">\parallel</TeX>, "\\parallel"],
       [<TeX key="angle">\angle</TeX>, "\\angle"],
       [<TeX key="bigtriangleup">\bigtriangleup</TeX>, "\\bigtriangleup"],
       [<TeX key="ps">▱</TeX>, "▱"],
       [<TeX key="bigodot">\bigodot</TeX>,  "\\bigodot"]
   ],
    "set": [
        [<TeX key="rightarrow">\rightarrow</TeX>, "\\rightarrow"],
        [<TeX key="leftrightarrow">\leftrightarrow</TeX>, "\\leftrightarrow"],
        [<TeX key="overarc">\overarc</TeX>, "\\overarc"],
        [<TeX key="parallel">\parallel</TeX>, "\\parallel"],
        [<TeX key="angle">\angle</TeX>, "\\angle"],
        [<TeX key="bigtriangleup">\bigtriangleup</TeX>, "\\bigtriangleup"],
        [<TeX key="bigodot">\bigodot</TeX>,  "\\bigodot"]
    ],
    "stat": [
        [<TeX key="mu">\mu</TeX>, "\\mu"],
        [<TeX key="sigma">\sigma</TeX>, "\\sigma"],
        [<TeX key="overline">{"\\overline{x}"}</TeX>,  input=>{
            input.latex("\\overline{x}")
        }],
        [<TeX key="x^i">x^i</TeX>, "x^i"],
        [<TeX key="x_i">x_i</TeX>, "x_i"],
        [<TeX key="x!">x!</TeX>, "x!"],
        [<TeX key="Sigma">\Sigma</TeX>,  "\\Sigma"]
    ],
    "greek": [
        [<TeX key="alpha">\alpha</TeX>, "\\alpha"],
        [<TeX key="beta">\beta</TeX>, "\\beta"],
        [<TeX key="gamma">\gamma</TeX>, "\\gamma"],
        [<TeX key="delta">\delta</TeX>, "\\delta"],
        [<TeX key="theta">\theta</TeX>, "\\theta"],
        [<TeX key="pi">\pi</TeX>, "\\pi"]
    ],
    "calculus": [
        [<TeX key="int">\int</TeX>, "\\int"],
        [<TeX key="int_{a}^{b}">{"\\int_{a}^{b}"}</TeX>, input=>{

            input.latex("\\int_{a}^{b}");
        }],
        [<TeX key="dx">dx</TeX>, "dx"],
        [<TeX key="frac{d}{dx}">{"\\frac{d}{dx}"}</TeX>, input=>{
            input.latex("\\frac{d}{dx}")
        }],
        [<TeX key="lim">{"\\lim_{x \\to \\infty}"}</TeX>, input=>{
            input.latex("\\lim_{x \\to \\infty}")
        }],
        [<TeX key="sum_{i=1}^{n}">{"\\sum_{i=1}^{n}"}</TeX>, input=>{
            input.latex("\\sum_{i=1}^{n}");

        }],
        [<TeX key="infty">\infty</TeX>,  "\\infty"]
    ]
};

class TexButtons extends Component{

    constructor(props){
        super(props);
        this.state = {
            buttonName: 'basic'
        }
    }

    render(){
        const buttonConfigs = [
            {key: 'basic', 'text': '常用'},
            {key: 'trig', text: '三角'},
            {key: 'logarithm', text: '指对'},
            {key: 'relation', text: '关系'},
            {key: 'geometry', text: '几何'},
            // {key: 'set', text: '集合'},
            // {key: 'group', text: '区间'},
            {key: 'stat', text: '统计'},
            {key: 'greek', text: '希腊字母'},
            {key: 'calculus', text: '微积分'}

        ];

        let buttonRows = {};

        for(const config of buttonConfigs){
            let name = config.key;
            let symbols = buttonSets[name];
            let buttonRow = _(symbols).map(symbol=>{
                return (
                    <button onClick={()=>{this.props.onInsert(symbol[1])}}
                            type="button"
                            key={symbol[0].key}
                            tabIndex={-1}
                            className="tex-button">
                        {symbol[0]}
                    </button>

                )
            });
            buttonRows[name] = <div className="tex-button-row">{buttonRow}</div>;
        }

        let buttonHeader = _(buttonConfigs).map(config =>{
            return <button
                        className={this.state.buttonName == config.key? "active": ""}
                        key={config.key}
                        onClick={()=>this.setState({buttonName: config.key})}
                        type="button"
                    >
                {config.text}
            </button>
        });

        return(
            <div className={`${this.props.className} preview-measure`}>
                <div className="tex-group">
                    {buttonHeader}
                </div>
                <div className="buttons">{buttonRows[this.state.buttonName]}</div>
            </div>
        )

    }
}



module.exports = TexButtons;


