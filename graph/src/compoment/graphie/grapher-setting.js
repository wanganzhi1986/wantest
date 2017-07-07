/**
 * Created by wangwei on 17/6/29.
 */
import React, { Component } from 'react';
import {Modal, Form, FormGroup, ControlLabel, FormControl, Checkbox} from 'react-bootstrap';
import {change} from '../../mixins/changeable';
import RangeInput from './range-input';
import _ from 'underscore';
import Util from '../../util/common';

class GrapherSetting extends  Component{

    constructor(props){
        super(props);
        this.state = this.stateFromProps(props)
    }

    change(...args){
        change.apply(this, args);
    }

    componentDidMount() {
        this.changeGraph = _.debounce(this.changeGraph, 300);
    }

    numSteps(range, step) {
        return Math.floor((range[1] - range[0]) / step);
    }


    //验证输入的范围
    validRange(range) {
        const numbers = _.every(range, function(num) {
            return _.isFinite(num);
        });
        if (! numbers) {
            return "范围必修是数字";
        }
        if (range[0] >= range[1]) {
            return "Range must have a higher number on the right";
        }
        return true;
    }

    validateStepValue(settings) {
        const { step, range, name, minTicks, maxTicks } = settings;

        if (! _.isFinite(step)) {
            return name + " must be a valid number";
        }
        const nSteps = this.numSteps(range, step);
        if (nSteps < minTicks) {
            return name + " is too large, there must be at least " +
                minTicks + " ticks.";
        }
        if (nSteps > maxTicks) {
            return name + " is too small, there can be at most " +
                maxTicks + " ticks.";
        }
        return true;
    }

    validSnapStep(step, range) {
        return this.validateStepValue({
            step: step,
            range: range,
            name: "Snap step",
            minTicks: 5,
            maxTicks: 60
        });
    }

    validGridStep(step, range) {
        return this.validateStepValue({
            step: step,
            range: range,
            name: "Grid step",
            minTicks: 3,
            maxTicks: 60
        });
    }

    validStep(step, range) {
        return this.validateStepValue({
            step: step,
            range: range,
            name: "Step",
            minTicks: 3,
            maxTicks: 20
        });
    }

    validateGraphSettings(range, step, gridStep, snapStep) {
        const self = this;
        let msg;
        const goodRange = _.every(range, function(range) {
            msg = self.validRange(range);
            return msg === true;
        });
        if (!goodRange) {
            return msg;
        }
        const goodStep = _.every(step, function(step, i) {
            msg = self.validStep(step, range[i]);
            return msg === true;
        });
        if (!goodStep) {
            return msg;
        }
        const goodGridStep = _.every(gridStep, function(gridStep, i) {
            msg = self.validGridStep(gridStep, range[i]);
            return msg === true;
        });
        if (!goodGridStep) {
            return msg;
        }
        const goodSnapStep = _.every(snapStep, function(snapStep, i) {
            msg = self.validSnapStep(snapStep, range[i]);
            return msg === true;
        });
        if (!goodSnapStep) {
            return msg;
        }
        return true;
    }

    changeLabel(i, e) {
        const val = e.target.value;
        const labels = this.state.labelsTextbox.slice();
        labels[i] = val;
        this.setState({ labelsTextbox: labels }, this.changeGraph);
    }

    changeRange(i, values) {
        const ranges = this.state.rangeTextbox.slice();
        ranges[i] = values;
        const step = this.state.stepTextbox.slice();
        const gridStep = this.state.gridStepTextbox.slice();
        const snapStep = this.state.snapStepTextbox.slice();
        const scale = Util.scaleFromExtent(ranges[i], this.props.box[i]);
        if (this.validRange(ranges[i]) === true) {
            step[i] = Util.tickStepFromExtent(
                ranges[i], this.props.box[i]);
            gridStep[i] = Util.gridStepFromTickStep(step[i], scale);
            snapStep[i] = gridStep[i] / 2;
        }
        this.setState({
            stepTextbox: step,
            gridStepTextbox: gridStep,
            snapStepTextbox: snapStep,
            rangeTextbox: ranges
        }, this.changeGraph);
    }

    changeStep(step) {
        this.setState({ stepTextbox: step }, this.changeGraph);
    }

    changeGridStep(gridStep) {
        this.setState({
            gridStepTextbox: gridStep,
            snapStepTextbox: _.map(gridStep, function(step) {
                return step / 2;
            })
        }, this.changeGraph);
    }

    changeGraph() {
        const labels = this.state.labelsTextbox;
        const range = _.map(this.state.rangeTextbox, function(range) {
            return _.map(range, Number);
        });
        const step = _.map(this.state.stepTextbox, Number);
        const gridStep = this.state.gridStepTextbox;
        const snapStep = this.state.snapStepTextbox;
        // const image = this.state.backgroundImage;

        // validationResult is either:
        //   true -> the settings are valid
        //   a string -> the settings are invalid, and the explanation
        //               is contained in the string
        // TODO(aria): Refactor this to not be confusing
        const validationResult = this.validateGraphSettings(range, step,
            gridStep, snapStep);

        if (validationResult === true) {  // either true or a string
            this.change({
                valid: true,
                labels: labels,
                range: range,
                step: step,
                gridStep: gridStep,
                snapStep: snapStep,
            });
        } else {
            this.change({
                valid: validationResult  // a string message, not false
            });
        }
    }

    stateFromProps(props) {
        return {
            labelsTextbox: props.labels,
            gridStepTextbox: props.gridStep,
            snapStepTextbox: props.snapStep,
            stepTextbox: props.step,
            rangeTextbox: props.range
        };
    }

    componentWillReceiveProps(nextProps) {
        // Make sure that state updates when switching
        // between different items in a multi-item editor.
        if (!_.isEqual(this.props.labels, nextProps.labels) ||
            !_.isEqual(this.props.gridStep, nextProps.gridStep) ||
            !_.isEqual(this.props.snapStep, nextProps.snapStep) ||
            !_.isEqual(this.props.step, nextProps.step) ||
            !_.isEqual(this.props.range, nextProps.range)) {
            this.setState(this.stateFromProps(nextProps));
        }
    }


    render(){
        return (
            <div>
                <div className="perseus-widget-row">
                    <div className="perseus-widget-left-col">
                        x Range
                        <RangeInput
                            value={this.state.rangeTextbox[0]}
                            onChange={(vals) => this.changeRange(0, vals)}
                        />
                    </div>
                    <div className="perseus-widget-right-col">
                        y Range
                        <RangeInput
                            value={this.state.rangeTextbox[1]}
                            onChange={(vals) => this.changeRange(1, vals)}
                        />
                    </div>
                </div>
                <div className="perseus-widget-row">
                    <div className="perseus-widget-left-col">
                        Tick Step
                        <RangeInput value= {this.state.stepTextbox}
                                    onChange = {(step)=>this.changeStep(step)} />
                    </div>
                    <div className="perseus-widget-right-col">
                        Grid Step
                        <RangeInput value= {this.state.gridStepTextbox}
                                    onChange = {(step)=>this.changeGridStep(step)} />
                    </div>
                </div>
            </div>
        )
    }
}

export default GrapherSetting