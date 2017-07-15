/**
 * Created by wangwei on 17/7/10.
 */
import React, {Component} from 'react'
const { Set, Rect, Text, Paper} = require('react-raphael');
import AxisContainer from './axes';
import Util from './util';
import './grapher.css'

class Grapher extends Component{

    constructor(props){
        super(props)
    }

    render(){
        const options = this.props.graph;
        const canvasData = Util.canvasData(options);
        const width = canvasData.width;
        const height = canvasData.height;
        const style = {position: 'relative'};
        return(
            <Paper width={width} height={height} container={{style: style, className:''}}>
                <AxisContainer options={this.props.graph}/>
            </Paper>

        )
    }
}

Grapher.defaultProps = {
    graph: {
        box: [400, 400],
        labels: ['x', 'y'],
        gridStep: [1, 1],
        step: [1, 1],
        snapStep: [1, 1],
        range: [[-10, 10], [-10, 10]],
        markings: 'graph',
        valid: true,
        isMobile: false
    },
    graphType: 'line'
};

export default Grapher