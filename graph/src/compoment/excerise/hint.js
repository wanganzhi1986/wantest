/**
 * Created by wangwei on 17/6/20.
 */

import React, { Component } from 'react';
import './question.css'
import { Button, ButtonToolbar, Glyphicon, Alert, Modal, Panel as BsPanel,
    Form, FormGroup, ControlLabel, FormControl, DropdownButton,
    MenuItem } from 'react-bootstrap';
import $ from 'jquery';
import Collapse, { Panel } from 'rc-collapse';
import TinyMCE from 'react-tinymce';
import 'rc-collapse/assets/index.css';
import TreeEditor from './tree.js';
import Select from 'react-select';
import 'react-select/dist/react-select.css';



function DeleteModal(props) {
    return (
        <Modal
            bsStyle='danger'
            show={props.isOpen}
            onHide={props.onRequestClose}
        >
            <Modal.Header closeButton>
                <Modal.Title>Delete base learner</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <p>Are you sure you want to delete this base learner setup?</p>
                <p>You will also lose all base learners that have been scored using this setup</p>
                <p><strong>This action is irreversible.</strong></p>
            </Modal.Body>
            <Modal.Footer>
                <Button bsStyle='danger' onClick={props.handleYes}>Yes</Button>
                <Button onClick={props.onRequestClose}>Cancel</Button>
            </Modal.Footer>
        </Modal>
    )
}


//生成知识点
class TopicGenerator extends Component{
    constructor(props){
        super(props);
        this.state = {
            topic: this.props.topic,
            showTopicModal : false
        }
    }

    //处理添加知识点后的回调函数
    handleAddTopicGenerator(topic){
        var option = {value: topic.name, label: topic.name, id:topic.id};
        this.props.handleOptionChange(option);
        this.setState({showTopicModal: false})
    }


    render(){

        var header = (
            <b>
                {this.props.topic}
                <a
                    className="DeleteButton"
                    onClick={(evt)=>{
                        evt.stopPropagation();
                        this.handleDeleteTopicGenerator(this.props.topic)
                    }}
                >
                    <Glyphicon glyph="remove" />
                </a>
            </b>
            ) ;
        return (
            <div>
                <Select multi
                        value={this.props.topic.values}
                        placeholder="添加知识点"
                        options={this.props.topic.options}
                        onChange={(value) => this.props.handleValueChange(value) } />

                <Glyphicon
                    glyph="plus"
                    style={{cursor:'pointer'}}
                    onClick={()=>this.setState({showTopicModal:true})}
                />
                <TopicModal
                    isOpen = {this.state.showTopicModal}
                    onRequestClose = {() => this.setState({showTopicModal: false})}
                    onAdd = {(topic) => this.handleAddTopicGenerator(topic)}
                />
            </div>
        )
    }
}

class TopicModal extends Component{
    constructor(props){
        super(props);
        this.state = {
            topic: {}
        }
    }

    handleSelectValue(event, treeId, treeNode){
        console.log("select topic name:", treeNode.name);
        console.log("当前对象是", this);
        this.setState({topic: treeNode})
    }

    render(){

        var setting = {
            data: {
                simpleData: {
                    enable: true
                }
            },
            callback:{
                //调用ztree提供点击事件的回调函数，
                onClick: this.handleSelectValue.bind(this)
            }
        };

        var zNodes =[
            { id:1, pId:0, name:"父节点1 - 展开", open:true},
            { id:11, pId:1, name:"父节点11 - 折叠"},
            { id:111, pId:11, name:"叶子节点111"},
            { id:112, pId:11, name:"叶子节点112"},
            { id:113, pId:11, name:"叶子节点113"},
            { id:114, pId:11, name:"叶子节点114"},
            { id:12, pId:1, name:"父节点12 - 折叠"},
            { id:121, pId:12, name:"叶子节点121"},
            { id:122, pId:12, name:"叶子节点122"},
            { id:123, pId:12, name:"叶子节点123"},
            { id:124, pId:12, name:"叶子节点124"},
            { id:13, pId:1, name:"父节点13 - 没有子节点", isParent:true},
            { id:2, pId:0, name:"父节点2 - 折叠"},
            { id:21, pId:2, name:"父节点21 - 展开", open:true},
            { id:211, pId:21, name:"叶子节点211"},
            { id:212, pId:21, name:"叶子节点212"},
            { id:213, pId:21, name:"叶子节点213"},
            { id:214, pId:21, name:"叶子节点214"},
            { id:22, pId:2, name:"父节点22 - 折叠"},
            { id:221, pId:22, name:"叶子节点221"},
            { id:222, pId:22, name:"叶子节点222"},
            { id:223, pId:22, name:"叶子节点223"},
            { id:224, pId:22, name:"叶子节点224"},
            { id:23, pId:2, name:"父节点23 - 折叠"},
            { id:231, pId:23, name:"叶子节点231"},
            { id:232, pId:23, name:"叶子节点232"},
            { id:233, pId:23, name:"叶子节点233"},
            { id:234, pId:23, name:"叶子节点234"},
            { id:3, pId:0, name:"父节点3 - 没有子节点", isParent:true}
        ];

        return (
            <Modal
                show={this.props.isOpen}
                onHide = {this.props.onRequestClose}
            >
                <Modal.Header closeButton>
                    <Modal.Title>选择知识点</Modal.Title>
                </Modal.Header>
                <Modal.Body>
                    <TreeEditor
                        setting={setting}
                        zNodes={zNodes}
                    />

                </Modal.Body>
                <Modal.Footer>
                    <Button
                        disabled={!this.state.topic}
                        bsStyle="primary"
                        onClick={()=>this.props.onAdd(this.state.topic)}
                    >
                        添加
                    </Button>
                    <Button onClick={()=>this.props.onRequestClose}>取消</Button>
                </Modal.Footer>
            </Modal>
        )
    }
}


//提示编辑器
class Hint extends Component{
    constructor(props){
        super(props);
        this.state = {

            // unsavedData:{
            //
            //     content: this.props.data.content,
            //     topic: {
            //         options: [],
            //         values: []
            //     }
            // },

            content: this.props.data.content,
            topic:{
                options: [],
                values: []
            },
            activeKey: '',
            showDeleteModal: false
        }
    }

    onActiveChange(activeKey){
        console.log(activeKey);
        this.setState({activeKey:activeKey})
    }

    //处理删除选择的提示项
    handleDeleteHint(){
        this.setState({showDeleteModal:true});
        this.props.deleteHint()
    }

    //添加一个选项，首先看是否已经添加到options中，如果没有，则添加，如果已经添加，
    //判断value是否在values,如果没有则将value添加的values中去
    handleTopicChange(option){
        this.setState((prevState) => {
            var topic = $.extend({}, prevState.topic);
            var newValue = option.value;
            var options = topic.options;
            var values = topic.values;
            var Idx = options.findIndex((x)=>x.id === option.id);
            var newValues = values + ',' + newValue;
            var newOptions = options.concat(option);
            if(Idx > -1){
                if(values.indexOf(newValue) < 0){
                    topic.values = newValues;
                }
            }
            else {
                topic.options = newOptions;
                topic.values = newValues;
            }
            return {topic: topic}
        })
    }

    handleTopicValueChange(values){
        this.setState((prevState)=>{
            var topic = $.extend({}, prevState.topic);
            topic.values = values;
            return {'topic': topic}
        })
    }


    handleDataChange(key, value){
        console.log("key is:"+key);
        console.log("value is:"+value);
        this.setState((prevState)=>{
            var newState = $.extend({}, prevState);
            newState.unsavedData[key] = value;
            return  newState
        })
    }


    render(){
        var header = <b>
            提示:
            <a className="DeleteButton"
               onClick={(evt)=>{
                   evt.stopPropagation();
                   this.setState({showDeleteModal: true})
               }
               }
            >
                <Glyphicon glyph="remove" />
            </a>
        </b>
        return(
            <div>
                <Collapse accordion={true} activeKey={this.state.activeKey}
                          onChange={(activekey)=>this.onActiveChange(activekey)}>
                    <Panel key={this.props.data.id} header={header}>
                        <TinyMCE
                            content={this.state.content}
                            onChange={(evt)=>this.handleDataChange('content', evt.target.getContent())}
                        />
                        <TopicGenerator
                            handleOptionChange = {(option)=> this.handleTopicChange(option)}
                            handleValueChange ={(value) => this.handleTopicValueChange(value)}
                            topic={this.state.topic}
                        />
                    </Panel>
                </Collapse>

                <DeleteModal
                    isOpen={this.state.showDeleteModal}
                    onRequestClose={() => this.setState({showDeleteModal: false})}
                    handleYes={() => this.handleDeleteHint(this.props.data.id)} />
            </div>
        )

    }
}

class ListHint extends Component{
    constructor(props){
        super(props);
    }

    getItems(){

        var items = this.props.hints.map((el, index)=>{
            return <Hint
                        key={el.id}
                        questionId={this.props.id}
                        data={el}
                        createHint={()=>{this.props.createHint()}}
                        deleteHint={()=>{this.props.deleteHint(el.id)}}
                        updateHint={(newData)=>{this.props.updateHint(el.id, newData)}}
                    />
        });

        return items

    }

    render(){
        return <div className="hint-editor">
            <h2>问题提示:</h2>
            {this.getItems()}
            <Button block onClick={this.props.createHint}>
                <Glyphicon glyph="plus" />
                {' 添加提示'}
            </Button>
        </div>
    }
}

export default ListHint