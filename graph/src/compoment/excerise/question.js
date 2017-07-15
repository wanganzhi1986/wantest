/**
 * Created by wangwei on 17/6/21.
 */
import React, { Component } from 'react';
import { Button, ButtonToolbar, Glyphicon, Alert, Modal, Panel as BsPanel,
    Form, FormGroup, ControlLabel, FormControl, DropdownButton, Radio, Checkbox,
    MenuItem } from 'react-bootstrap';
import $ from 'jquery';
import ListHint from './hint.js';
import MathInput from '../math/math-input';
import Editor from '../editor/index';
import GrapherEditor from '../editor/plugins/graph/grapher-editor';


class Question extends Component{
    constructor(props){
        super(props);
        this.state = {
            hints: [
                {id:1, content:'若干个', topic:'三角'},
                {id:2, content:'三角形的面积', topic:'面积'}
            ],
            answer: '',
            selectType: '',
            description: ''
        }
    }

    createQuestionHint(){
        console.log("创建问题提示");
        this.setState((prevState)=>{
            var hints = prevState.hints.slice();
            var newId = hints.length + 1;
            hints.push({id:newId, content: '', topic:''});
            return {hints}
        })

    }

    deleteQuestionHint(hid){
        console.log("删除问题提示"+hid);
        this.setState((prevState)=>{
            var hints = prevState.hints.slice();
            var idx = hints.findIndex((x) =>x.id === hid);
            if(idx > -1){
                hints.splice(idx, 1);
            }
            return {hints}
        })
    }

    updateQuestionHint(hid, newData){
        console.log("更新问题提示列表"+ hid +":" + newData);
        this.setState((prevState)=>{
            var hints = prevState.hints.slice();
            var Idx = hints.findIndex((x)=>x.id===hid);
            hints[Idx] = newData;
            return {hints}
        })
    }

    handleDataChange(key, value){
        console.log("key is:"+key);
        console.log("value is:"+value);
        this.setState((prevState)=>{
            var newState = $.extend({}, prevState);
            newState[key] = value;
            return  newState
        })
    }


    render(){
        return (
            <div>
                <QuestionType
                    selected={this.state.selectType}
                    handleTypeChange={(key, value)=>{this.handleDataChange(key, value)}}
                />
                {/*<QuestionDescription*/}
                    {/*description={this.state.description}*/}
                    {/*handleDescriptionChange={(key, value)=>{this.handleDataChange(key, value)}}*/}

                {/*/>*/}
                <QuestionAnswer
                    answer={this.state.answer}
                    handleAnswerChange={(key, value)=>{this.handleDataChange(key, value)}}
                />
                <ListHint
                    hints={this.state.hints}
                    createHint={()=>this.createQuestionHint()}
                    deleteHint={(hid)=>this.deleteQuestionHint(hid)}
                    updateHint={(hid, source)=>this.updateQuestionHint(hid, source)}
                />
                <Editor
                    placeholder="开始输入内容"
                />
                <GrapherEditor/>
            </div>
        )
    }
}

class QuestionType extends Component{
    constructor(props){
        super(props);
    }

    render(){
        return(
            <div className="question-type">
                <FormGroup>
                    <Radio name="questionType" value="xz" inline
                           onChange={(evt)=>{this.props.handleTypeChange('type', evt.target.value)}}
                            >
                        选择题</Radio>
                    <Radio name="questionType" value="tk" inline
                           onChange={(evt)=>{this.props.handleTypeChange('type', evt.target.value)}}
                    >填空题</Radio>
                    <Radio name="questionType" value="jd" inline
                           onChange={(evt)=>{this.props.handleTypeChange('type', evt.target.value)}}
                    >解答题</Radio>
                </FormGroup>
            </div>

        )

    }
}

//输入答案组件
class QuestionAnswer extends Component{
    constructor(props){
        super(props);
    }

    render(){
        return (
            <div>
               <span>输入答案:</span>
                <MathInput
                    value={this.props.answer}
                    convertDotToTimes={true}
                    onChange = {(value)=> this.props.handleAnswerChange('answer', value)}
                />
            </div>
        )
    }
}

export default Question




