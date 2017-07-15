/**
 * Created by wangwei on 17/7/13.
 */
import React, {Component} from 'react'
import Button from '../../ui/button';
import Dropdown from '../../ui/dropdown';
import {GraphConfigs} from './constant';


class ToolBar extends Component{
    constructor(props){
        super(props);
        this.state = {
            visible: ''
        }
    }

    getControls(groups){
        if (!Array.isArray(groups[0])) {
            groups = [groups];
        }
        return groups.map(function(controls) {
                return controls.map(function (control) {
                    let name= control.name;
                    let config = GraphConfigs[name];
                    let title = control.title;
                    let style = control.style;
                    if (!config) return;
                    if( config.length > 1){
                        return <Dropdown
                                key={name}
                                style={style}
                                configs={config}
                                title={title}
                                name={name}
                                onChange={(ele)=> this.props.onChange(ele)}
                                open={this.state.visible == name}
                                onVisibleChange={(name)=>{
                                    this.setState({visible: name})
                                }}
                            />

                    }
                    else {
                        return <Button
                                key={name}
                                style={style}
                                configs={config}
                                name={name}
                                title={title}
                                onChange={(ele)=>this.props.onChange(ele)}
                             />
                    }
                }.bind(this));
            }.bind(this)
        );
    }


    render(){
        let buttonControls = this.getControls(this.props.toolBarConfigs);
        return (
            <div className="toolbar">
                {buttonControls}
            </div>
        )
    }
}

ToolBar.defaultProps = {
    toolBarConfigs:[
        {
            name: "point",
            title: '点'
        },
        {
            name: "line",
            title: '直线'
        },
        {
            name: "polygon",
            title: '多边形'
        },
        {
            name: 'circle',
            title: '圆形'
        },
        {
            name: 'func',
            title: '常用函数'
        },
        {
            name: 'text',
            title: '文本'
        },
        {
            name: 'setting',
            title: '设置',
            style: {
                'float': 'right',
                'marginRight': '30px'
            }
        }
    ]
};

export default ToolBar