/**
 * Created by wangwei on 17/6/19.
 */
import React, { Component } from 'react';
import { Button, ButtonToolbar, Glyphicon, Alert, Modal, Panel as BsPanel,
    Form, FormGroup, ControlLabel, FormControl, DropdownButton,
    MenuItem } from 'react-bootstrap';

import { Radio, RadioGroup } from 'react-bootstrap-validation';
import FaCheck from 'react-icons/lib/fa/check';
import FaSpinner from 'react-icons/lib/fa/spinner';
import ListHint from 'hint.js'


//练习组件
class Excerise extends Component{
    constructor(props){
        super(props);
        this.state = {

        }
    }

    render(){
        return(
            <div>
                <ExceriseType
                />
                <Answer
                />
                <ListHint
                />

            </div>
        )


    }
}


class ContentEditor extends Component{

    constructor(props) {
        super(props);
        this.state = {
            baseLearners: [],
        };
    };


    render(){
        return <div className=""></div>
    }
}


//练习类型组件
class ExceriseType extends Component{
    constructor(props){
        super(props);
        this.state = {

        }
    };

    render(){

    }
}

//选择题编辑器
class ChoiceExceriseEditor extends Component{
    constructor(props){
        super(props);
        this.state = {

        }
    }

    render(){

    }
}

//填空题编辑器
class FillExceriseEditor extends Component{
    constructor(props){
        super(props);
        this.state = {};

    }

    render(){

    }
}

class Answer extends Component{
    constructor(props){
        super(props);
        this.state = {}
    }

    render(){
        return <div>
            <p>答案:</p>
            <ContentEditor/>
            <ListHint/>
        </div>
    }
}

