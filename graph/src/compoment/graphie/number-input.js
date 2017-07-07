/**
 * Created by wangwei on 17/6/29.
 */
import React, { Component } from 'react';
import ReactDOM from 'react-dom'
import classNames from 'classnames';
import Util from '../../util/common';
import _ from 'underscore';
import $ from 'jquery';

let firstNumericalParse = Util.firstNumericalParse;
let captureScratchpadTouchStart = Util.captureScratchpadTouchStart;

import kmath from 'kmath';

import {KhanMath} from '../../util/math';

let toNumericString = KhanMath.toNumericString;
let getNumericFormat = KhanMath.getNumericFormat;
let knumber = kmath.number;

class NumberInput extends Component{
    constructor(props){
        super(props);
        this.state = {
            format: ''
        }
    }

    render() {

        var classes = classNames({
            "number-input": true,
            "invalid-input": !this._checkValidity(this.props.value),
            "mini": this.props.size === "mini",
            "small": this.props.size === "small",
            "normal": this.props.size === "normal"
        });
        if (this.props.className != null) {
            classes = classes + " " + this.props.className;
        }


        return <input
            className={classes}
            type="text"
            ref={(input)=>{this.numberInput=input}}
            onChange={(e)=>this._handleChange(e)}
            onFocus={()=>this.focus()}
            onBlur={(e)=>this._handleBlur(e)}
            onKeyPress={(e)=>this._handleBlur(e)}
            onKeyDown={(e)=>this._onKeyDown(e)}
            defaultValue={toNumericString(this.props.value, this.state.format)}
            value={undefined}
        />;
    }

    componentDidUpdate(prevProps) {
        let val = this.parseInputValue(this.numberInput.value);
        if (!knumber.equal(val, this.props.value)) {
            // let newValue = toNumericString(this.props.value, this.state.format);
            this._setValue(val, this.state.format)
        }
    }

    _setValue(val, format) {
        $(this.numberInput).val(toNumericString(val, format));
    }



    /* Return the current string value of this input */
    getStringValue() {
        return this.numberInput.value.toString();
    }

    parseInputValue(value) {
        if (value === "") {
            const placeholder = this.props.placeholder;
            return _.isFinite(placeholder) ? +placeholder : null;
        } else {
            let result = firstNumericalParse(value);
            // console.log("是否是数字:", Number.isFinite(result));
            return Number.isFinite(result) ? result : this.props.value;
            // return result;
        }
    }

    /* Set text input focus to this input */
    focus() {
        this.numberInput.focus();
        this._handleFocus();
    }

    blur(){
        this.numberInput.blur();
        this._handleBlur();
    }

    setSelectionRange(selectionStart, selectionEnd) {
        this.numberInput.setSelectionRange(selectionStart, selectionEnd);
    }

    getSelectionStart() {
        return ReactDOM.findDOMNode(this).selectionStart;
    }

    getSelectionEnd() {
        return ReactDOM.findDOMNode(this).selectionEnd;
    }

    _checkValidity(value) {
        if (value == null) {
            return true;
        }

        var val = firstNumericalParse(value);
        var checkValidity = this.props.checkValidity;

        return _.isFinite(val) && checkValidity(val);
    }

    getValue() {
        return this.parseInputValue(this.numberInput.value);
    }

    _handleChange(e) {
        let text = e.target.value;
        let value = this.parseInputValue(text);
        let format = getNumericFormat(text);
        this.props.onChange(value);
        if (format) {
            // this.props.onFormatChange(value, format);
            this.setState({format: format});
        }
    }

    _handleFocus() {
        if (this.props.onFocus) {
            this.props.onFocus();
        }
    }

    _handleBlur(e) {
        // Only continue on blur or "enter"
        if (e && e.type === "keypress" && e.keyCode !== 13) {
            return;
        }
        this._setValue(this.props.value, this.state.format);
        if (this.props.onBlur) {
            this.props.onBlur();
        }
    }

    _onKeyDown(e) {
        if (this.props.onKeyDown) {
            this.props.onKeyDown(e);
        }

        if (!this.props.useArrowKeys ||
            !_.contains(["ArrowUp", "ArrowDown"], e.key)) {
            return;
        }

        var val = this.getValue();
        if (val !== Math.floor(val)) {
            return; // bail if not an integer
        }

        if (e.key === "ArrowUp") {
            val = val + 1;
        } else if (e.key === "ArrowDown") {
            val = val - 1;
        }

        if (this._checkValidity(val)) {
            this.props.onChange(val);
        }
    }

}

export default NumberInput;














