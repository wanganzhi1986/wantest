/**
 * Created by wangwei on 17/7/13.
 */
import React, {Component} from 'react';

class ComboBox extends Component{
    constructor(props){
        super(props);
        this.state = {
            show:false,
            position:{
                x:0,
                y:0
            }
        }
    }
    componentDidMount(){
        window.addEventListener("click",this.close.bind(this));
    }
    componentWillUnmount(){
        window.removeEventListener("click",this.close.bind(this));
    }
    open(position){
        this.setState({
            show:true,
            position:position
        })
    }
    close(){
        if(!this.state.show) return;
        this.setState({
            show:false
        })
    }
    toggle(position){
        this.setState({
            show: !this.state.show,
            position: position
        })
    }

    _handleSelect(e, config){
        e = e || event;
        let target = e.target || e.srcElement;
        let value = target.getAttribute('data-value');
        if(this.state.handle){
            this.state.handle(e,value);
        }
        if(e.stopPropagation){
            e.stopPropagation();
        }else{
            e.cancelBubble = true;
        }
        this.close();

    }

    render(){
        let { className, style, configs} = this.props;
        style = style || {};

        let subStyle = {};
        if(!this.state.show){
            subStyle["display"] = "none";
        }else{
            subStyle["display"] = "";
        }
        if(this.state.position){
            subStyle["left"] = this.state.position.x;
            subStyle["top"] = this.state.position.y;
        }
        const name = this.props.key;
        const subItems = configs.map(function (config) {
            return (
                    <li key={config.key} onClick={(e, config)=>this._handleSelect(e, config)}>
                        <span>{config.title}</span>
                    </li>
                )
        });

        return (
            <div style={style} className={"combobox" + (className ? " " + className : "")}>
                <div className="toolbar-button">{name}</div>
                <div className="sub_menu" style={subStyle}>
                    <ul>
                        {subItems}
                    </ul>
                </div>
            </div>)
    }
}

export default ComboBox;
