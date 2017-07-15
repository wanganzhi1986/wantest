/**
 * Created by wangwei on 17/7/10.
 */
import React, {Component} from 'react'
import Util from './util';
import GraphUtil from './graph-util';
import {Path, Set, Raphael} from 'react-raphael';
import {KhanMath, TexUtil}  from "../../../../util/math";
import ReactDOM from 'react-dom';
import $ from 'jquery';
import extend from 'extend';

const processMath = TexUtil.processMath;

class Grid extends React.Component{
    constructor(props){
        super(props);
    }
    render(){
        console.log('grid is:')
        const options = this.props.options;
        const gridData = Util.gridData(options);
        const xGridPoints = gridData.xGridPoints;
        const yGridPoints = gridData.yGridPoints;
        const attr = gridData.attr;
        return (<Set>
            {
                xGridPoints.map(function(points,pos) {
                    let path = GraphUtil.svgPath(points);
                    return <Path d={path} key={'x-grid-'+ pos} attr={attr}/>
                }
                    )
                }
            {

                yGridPoints.map(function(points,pos){
                    let path = GraphUtil.svgPath(points);
                    return <Path d={path} key={'y-grid-'+pos} attr={attr}/>
                })
            }
        </Set>)
    }
}

class Axis extends Component{

    constructor(props){
        super(props);
        this.state = {
            xArrow: {},
            yArrow: {}
        }
    }

    componentDidMount(){

        window.onload = function () {
            this.handleArrowPath(this.xInput, 'x');
            this.handleArrowPath(this.yInput, 'y');
        }.bind(this);

    }

    handleArrowPath(input, tag){
        const path = input.getElement();
        console.log('path is:', path);
        const strokeWidth = path.attr("stroke-width");
        console.log('stroke width is:', path.attr())
        const s = 0.6 + 0.4 * strokeWidth;
        const l = path.getTotalLength();
        const subPath = "M-3 4 C-2.75 2.5 0 0.25 0.75 0C0 -0.25 -2.75 -2.5 -3 -4";
        const end = path.getPointAtLength(l - 0.4);
        const almostTheEnd = path.getPointAtLength(l - 0.75 * s);
        const angle = Math.atan2(
                end.y - almostTheEnd.y,
                end.x - almostTheEnd.x) * 180 / Math.PI;
        const rotateData = {angle: angle, cx: 0.75, cy: 0};
        const scaleData = {sx:s, sy:s, cx:0.75, cy: 0};
        const translateData = {x:almostTheEnd.x, y: almostTheEnd.y};
        const attrData = {"stroke-linejoin": "round",
                           "stroke-linecap": "round",
                            stroke: "#000000",
                            strokeWidth: 2,
                            opacity: 1

        };
        const arrowData = {
            path: subPath,
            rotateData: rotateData,
            scaleData: scaleData,
            translateData: translateData,
            attrData: attrData
        };
        if (tag=='x'){
            this.setState({xArrow: arrowData})
        }

        if(tag == 'y'){
            this.setState({yArrow: arrowData})
        }

    }

    render(){

        const options = this.props.options;
        const axisData = Util.axisData(options);
        const xAxisPoint = axisData.xAxisPoint;
        const yAxisPoint = axisData.yAxisPoint;
        const attr = axisData.attr;
        let xPath = GraphUtil.svgPath(xAxisPoint);
        let yPath = GraphUtil.svgPath(yAxisPoint);
        return (
            <Set>
                <Path d={xPath} attr={attr} ref={(input)=>this.xInput=input}/>
                <Path d={this.state.xArrow.path}
                      rotate={this.state.xArrow.rotateData}
                      scale={this.state.xArrow.scaleData}
                      translate={this.state.xArrow.translateData}
                      attr={this.state.xArrow.attrData}/>
                <Path d={yPath} attr={attr} ref={(input)=>this.yInput=input}/>
                <Path d={this.state.yArrow.path}
                      rotate={this.state.yArrow.rotateData}
                      scale={this.state.yArrow.scaleData}
                      translate={this.state.yArrow.translateData}
                      attr={this.state.yArrow.attrData}/>
            </Set>

        )
    }
}

const labelDirections = {
    "center": [-0.5, -0.5],
    "above": [-0.5, -1.0],
    "above right": [0.0, -1.0],
    "right": [0.0, -0.5],
    "below right": [0.0, 0.0],
    "below": [-0.5, 0.0],
    "below left": [-1.0, 0.0],
    "left": [-1.0, -0.5],
    "above left": [-1.0, -1.0],
};

class Label extends Component{

    constructor(props){
        super(props);
    }

    setLabelMargins(el, size, direction) {
        const $span = el;
        if (typeof direction === "number") {
            const x = Math.cos(direction);
            const y = Math.sin(direction);

            const scale = Math.min(
                size[0] / 2 / Math.abs(x),
                size[1] / 2 / Math.abs(y));

            $span.css({
                marginLeft: (-size[0] / 2) + x * scale,
                marginTop: (-size[1] / 2) - y * scale,
            });
        } else {
            const multipliers = labelDirections[direction || "center"];
            $span.css({
                marginLeft: Math.round(size[0] * multipliers[0]),
                marginTop: Math.round(size[1] * multipliers[1]),
            });
        }
    };

    componentDidMount(){
        let point = this.props.point;
        let pad = this.props.pad;
        let text = this.props.text;
        let direction = this.props.direction;
        let el = this.axisLabel;
        let $el = $(el);
        $el.css({
            position: 'absolute',
            padding: (pad != null ? pad : 7) + "px",
            left: point[0],
            top: point[1]
        });

        processMath(el, text, false, function () {
            const width = this.axisLabel.scrollWidth;
            const height = this.axisLabel.scrollHeight;
            this.setLabelMargins($el, [width, height], direction);
        }.bind(this)
        )


    }

    render(){
        return (
            <span className="axis-label"
                  ref={(input)=> this.axisLabel=input}>
            </span>
        )

    }
}


class AxisLabels extends Component{
    render(){
        const options = this.props.options;
        const labelData = Util.labelData(options);
        const xLabelPoints = labelData.xLabelPoint;
        const yLabelPoints = labelData.yLabelPoint;
        return (
            <Set>
                {xLabelPoints.map(function (label, pos) {
                        return <Label
                                    key={'xlabel-'+pos}
                                    point={label.point}
                                    text={label.text}
                                    direction={label.direction}
                        />

                    })
                }

                {
                    yLabelPoints.map(function (label, pos) {
                            return <Label
                                key={'ylabel-'+pos}
                                point={label.point}
                                text={label.text}
                                direction={label.direction}
                            />
                }
                    )
                }
            </Set>
        )

    }
}

class AxisTicks extends Component{
    render(){
        const options = this.props.options;
        const tickData = Util.tickData(options);
        const xTickPoints = tickData.xTickPoints;
        const yTickPoints = tickData.yTickPoints;
        const attr = tickData.attr;
        return (
            <Set>
                {
                    xTickPoints[0].map(function (points, pos) {
                        let path = GraphUtil.svgPath(points);

                        return <Path d={path} key={'x-tick-plus-'+ pos} attr={attr}/>
                    })
                }
                {
                    xTickPoints[1].map(function (points, pos) {
                        let path = GraphUtil.svgPath(points);
                        return <Path d={path} key={'x-tick-minus-'+ pos} attr={attr}/>
                    })
                }
                {
                    yTickPoints[0].map(function (points, pos) {
                        let path = GraphUtil.svgPath(points);
                        return <Path d={path} key={'y-tick-plus-'+ pos} attr={attr}/>
                    })
                }
                {
                    yTickPoints[1].map(function (points, pos) {
                        let path = GraphUtil.svgPath(points);
                        return <Path d={path} key={'y-tick-minus-'+pos} attr={attr}/>
                    })
                }
            </Set>
        )
    }
}


class Arrow extends Component {

    render(){
        console.log('arrows is:');
        const points = this.props.points;
        const path = Raphael
        const strokeWidth = this.props.strokeWidth;
        const s = 0.6 + 0.4 * strokeWidth;
        const l = path.getTotalLength();
        const subPath = "M-3 4 C-2.75 2.5 0 0.25 0.75 0C0 -0.25 -2.75 -2.5 -3 -4";
        const end = path.getPointAtLength(l - 0.4);
        const almostTheEnd = path.getPointAtLength(l - 0.75 * s);
        const angle = Math.atan2(
                end.y - almostTheEnd.y,
                end.x - almostTheEnd.x) * 180 / Math.PI;
        const rotateData = {angle: angle, cx: 0.75, cy: 0};
        const scaleData = {sx:s, sy:s, cx:0.75, cy: 0};
        const translateData = {x:almostTheEnd.x, y: almostTheEnd.y};
        const attrData = {"stroke-linejoin": "round", "stroke-linecap": "round"};

        return <Path d={subPath}
                     rotate={rotateData}
                     scale={scaleData}
                     translate={translateData}
                     attr={attrData}/>
    }
}


class AxisContainer extends Component{

    render(){
        console.log("开始axis");
        return(
            <Set>
                <Grid options={this.props.options}/>

                <Axis options={this.props.options}/>
                <AxisTicks options={this.props.options}/>

                <AxisLabels options={this.props.options}/>



            </Set>
        )
    }


}

export default AxisContainer