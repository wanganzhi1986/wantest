/**
 * Created by wangwei on 17/6/29.
 */

import React, { Component } from 'react';
import NumberInput from './number-input';

var truth = () => true;

/* A minor abstraction on top of NumberInput for ranges
 *
 */

 class RangeInput extends Component{

     constructor(props){
         super(props);
         this.state = {
             placeholder: [null, null]
         }
     }


    render() {
        let value = this.props.value;
        let checkValidity = this.props.checkValidity || truth;

        return <div className="range-input">
            <NumberInput
                value={value[0]}
                checkValidity={val => checkValidity([val, value[1]])}
                onChange={(newVal)=>this.onChange(0, newVal)}
                placeholder={this.state.placeholder[0]} />
            <NumberInput
                value={value[1]}
                checkValidity={val => checkValidity([value[0], val])}
                onChange={(newVal) => this.onChange(1, newVal)}
                placeholder={this.state.placeholder[1]} />
        </div>;
    }

    onChange(i, newVal) {
        let value = this.props.value;
        if (i === 0) {
            this.props.onChange([newVal, value[1]]);
        } else {
            this.props.onChange([value[0], newVal]);
        }
    }

}

export default RangeInput
