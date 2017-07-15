/**
 * Created by wangwei on 17/7/13.
 */
import React, {Component} from 'react'
import './button.css'

class Button extends Component{
    constructor(props){
        super(props);
    }


    _handleClick(config){
        this.props.onChange(config);
    }

    render(){
        let { className, style } = this.props;
        const config = this.props.configs[0];
        return (
            <div className={"button" + (className ? " " + className : "")} style={style}>
                <button value={config.key}
                        onClick={(config)=>this._handleClick(config)}
                >
                    {config.title}
                </button>
            </div>
        )
    }
}

export default Button;