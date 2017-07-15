/**
 * Created by wangwei on 17/7/13.
 */
import React, {Component} from 'react'
import './button.css'

class Dropdown extends Component{
    constructor(props){
        super(props);
        this.state = {
            show:false,
            selected: '',
            position:{
                x:0,
                y:0
            }
        };
    }

    componentDidMount(){
        window.addEventListener("click",this.close.bind(this));
    }

    componentWillUnmount(){
        window.removeEventListener("click",this.close.bind(this));
    }

    componentWillReceiveProps(nextProps){
        if (this.props.open !== nextProps.open){
            this.setState({show: nextProps.open})
        }

    }

    open(position){
        this.setState({
            show:true,
            position:position
        })
    }
    close(e){
        if(!this.state.show) return;
        this.setState({
            show:false
        })
    }
    toggle(e){

        //
        if (this.state.show){
            this.setState({
                show:!this.state.show,
            });
        }
        else {
            this.props.onVisibleChange(this.props.name)
        }

        e.stopPropagation()
    }

    handleClick(e, config){
        // if ( e && e.preventDefault ){
        //     e.preventDefault();
        // }
        this.setState({
            show: !this.state.show
        });
        this.props.onChange(config);
        e.stopPropagation();
    }

    render(){
        let { className, style, configs, name, title} = this.props;
        style = style || {};
        // style['position'] = 'relative';

        let subStyle = {};
        // console.log('修改状态是:', this.props.open)
        if(!this.state.show){
            subStyle["display"] = "none";
        }else{
            subStyle["display"] = "";
        }

        // if(this.state.position){
        //     subStyle["left"] = this.state.position.x;
        //     subStyle["top"] = this.state.position.y;
        // }
        // subStyle['position'] = 'absolute';
        const subItems = configs.map(function (config) {
            return (
                <li className='subItem' key={config.key} onClick={(e, config)=>this.handleClick(e, config)}>
                    <div className="itemLabel">{config.title}</div>
                </li>
            )
        }.bind(this));


        return (
            <div style={style} className={"dropdown" + (className ? " " + className : "")}>
                <div className="toolbar-button">
                    <button type='button' value={name} onClick={this.toggle.bind(this)}>{title}</button>
                </div>
                <div className="subItems" style={subStyle}>
                    <div className="item-arrow">
                        <div className="item-arrow-shadow"></div>
                    </div>
                    <ul className="subItemContent">
                        {subItems}
                    </ul>
                </div>
            </div>)
    }
}

export default Dropdown;
