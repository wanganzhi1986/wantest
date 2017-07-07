/**
 * Created by wangwei on 17/6/23.
 */

import React, { Component } from 'react';
// import clone from 'lodash/lang/clone';

class TreeEditor extends Component{
    constructor(props){
        super(props);
        this.state = {}
    }

    componentWillMount() {
        this.id = this.id || this.props.id || "tree";
    }

    componentDidMount(){
        const setting = this.props.setting;
        const zNodes =this.props.zNodes;
        this._init(setting, zNodes);
    }

    _init(setting, zNodes){
        var zId = "#" + this.id;
        $(document).ready(function () {
            $.fn.zTree.init($(zId), setting, zNodes);
        })
    }


    render(){
        return (
            <div className={this.props.className}>
                <ul id={this.id} className="ztree">''</ul>
            </div>)
    }
}

export default TreeEditor