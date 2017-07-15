/**
 * Created by wangwei on 17/7/10.
 */
import React, {Component} from 'react';
import QuillEditor from './quill-editor';

class Editor extends Component{
    constructor(props){
        super(props);
        this.state = {editorHtml: ''}
    }

    handleChange (html) {
        this.setState({ editorHtml: html });
    }

    render(){
        return <div className="editor">
            <QuillEditor
                theme="snow"
                modules={Editor.modules}
                formats={Editor.formats}
                onChange={(editorHtml)=> this.handleChange(editorHtml)}
                placeholder={this.props.placeholder}
            />
        </div>
    }
}

Editor.modules =  {
    toolbar: [
        [{ 'header': [1, 2, false] }],
        ['bold', 'italic', 'underline','strike', 'blockquote'],
        [{'list': 'ordered'}, {'list': 'bullet'}, {'indent': '-1'}, {'indent': '+1'}],
        ['link', 'image', 'formula'],
        ['clean']
    ],
};

Editor.formats = [
    'header',
    'bold', 'italic', 'underline', 'strike', 'blockquote',
    'list', 'bullet', 'indent',
    'link', 'image'
];

export default Editor;