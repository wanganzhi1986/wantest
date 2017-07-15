/**
 * Created by wangwei on 17/7/13.
 */
import React, {Component} from 'react'

class Button extends Component{
    constructor(props){
        super(props);
    }

    render(){
        let { className, style, ...props } = this.props;

        return (
            <div className="button">
                <button style={style}></button>
            </div>
        )
    }
}