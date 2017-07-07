/**
 * Created by wangwei on 17/6/29.
 */
import React, { Component } from 'react';
import GrapherButtonGroup from './grapher-button';
import Grapher from './grapher';
import _ from 'underscore';

class GrapherEditor extends Component {

    constructor(props) {
        super(props);
        this.state = {
            graphSetting: {
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

        }
    };

        render(){
            return (
                <div className="grapher-container">
                    <div className="button-group">
                        <GrapherButtonGroup
                            graphSetting={this.state.graphSetting}
                            graphType={this.state.graphType}
                            onChange={(value)=>{
                                this.setState((prevState)=>{
                                     _.extend({}, prevState, value);
                                })
                            }}
                        />
                    </div>
                    <div className="grapher-show">
                        <Grapher
                            graph={this.state.graphSetting}
                        />
                    </div>
                </div>
            )

        }
}

export  default  GrapherEditor