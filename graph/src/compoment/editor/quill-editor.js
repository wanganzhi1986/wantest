/**
 * Created by wangwei on 17/7/8.
 */
import React, {Component} from 'react';
import ReactDOM from 'react-dom';
import Quill from 'quill';
import 'quill/dist/quill.core.css';
import 'quill/dist/quill.snow.css';
import isEqual from 'lodash/isequal';


class QuillEditor extends Component{


    constructor(props){
        super(props);
        this.state = {
            generation: 0
        };
        this.registerPlugins(this.props.plugins)
    }

    registerPlugins(plugins){
        if (plugins){
            Object.keys(plugins).forEach((typeName)=>{
                Object.keys(plugins[typeName]).forEach((plugName)=>{
                    let path = typeName + '/' + plugName;
                    let target = plugins[typeName][plugName];
                    Quill.register(path, target);
                })
            })
        }
    }

    componentDidMount () {
        const self = this;
        this.editor = this.createEditor(
            this.getEditingArea(),
            this.getEditorConfig()
        );
        this.editor.root.addEventListener('compositionend', () => {
            self.editor.selection.cursor.restore();
        });

        // Restore editor from Quill's native formats in regeneration scenario
        if (this.quillDelta) {
            this.editor.setContents(this.quillDelta);
            this.editor.setSelection(this.quillSelection);
            this.editor.focus();
            this.quillDelta = null;
            this.quillSelection = null;
            return;
        }
        if (this.state.value) {
            this.setEditorContents(this.editor, this.state.value);
        }
    }

    componentWillReceiveProps (nextProps, nextState) {
        const editor = this.editor;

        // If the component is unmounted and mounted too quickly
        // an error is thrown in setEditorContents since editor is
        // still undefined. Must check if editor is undefined
        // before performing this call.
        if (!editor) return false;

        // Update only if we've been passed a new `value`.
        // This leaves components using `defaultValue` alone.
        if ('value' in nextProps) {
            // NOTE: Seeing that Quill is missing a way to prevent
            //       edits, we have to settle for a hybrid between
            //       controlled and uncontrolled mode. We can't prevent
            //       the change, but we'll still override content
            //       whenever `value` differs from current state.
            if (!isEqual(nextProps.value, this.getEditorContents())) {
                this.setEditorContents(editor, nextProps.value);
            }
        }

        // We can update readOnly state in-place.
        if ('readOnly' in nextProps) {
            if (nextProps.readOnly !== this.props.readOnly) {
                this.setEditorReadOnly(editor, nextProps.readOnly);
            }
        }

        // If we need to regenerate the component, we can avoid a detailed
        // in-place update step, and just let everything rerender.
        if (this.shouldComponentRegenerate(nextProps, nextState)) {
            return this.regenerate();
        }
        return false;
    }

    shouldComponentRegenerate (nextProps) {
        // Whenever a `dirtyProp` changes, the editor needs reinstantiation.
        return QuillEditor.dirtyProps.some(
            prop =>
                // Note that `isEqual` compares deeply, making it safe to perform
                // non-immutable updates, at the cost of performance.
                !isEqual(nextProps[prop], this.props[prop])
        );
    }

    componentWillUnmount () {
        const editor = this.getEditor();
        if (editor) {
            this.unhookEditor(editor);
            this.editor = null;
        }
    }
    shouldComponentUpdate (nextProps, nextState) {
        // If the component has been regenerated, we already know we should update.
        if (this.state.generation !== nextState.generation) {
            return true;
        }

        // Compare props that require React updating the DOM.
        return QuillEditor.cleanProps.some(
            prop =>
                // Note that `isEqual` compares deeply, making it safe to perform
                // non-immutable updates, at the cost of performance.
                !isEqual(nextProps[prop], this.props[prop])
        );
    }

    getEditorConfig () {
        return {
            bounds: this.props.bounds,
            formats: this.props.formats,
            modules: this.props.modules,
            placeholder: this.props.placeholder,
            readOnly: this.props.readOnly,
            theme: this.props.theme
        };
    }

    getEditor () {
        return this.editor;
    }

    getEditingArea () {
        return ReactDOM.findDOMNode(this.editingArea); // eslint-disable-line
    }

    getEditorContents () {
        return this.state.value;
    }

    getEditorSelection () {
        return this.state.selection;
    }
    convertHtml (html) {
        if (Array.isArray(html)) return html;
        return this.editor.clipboard.convert(
            `<div class='ql-editor' style="white-space: normal;">${html}<p><br></p></div>`
        );
    }

    regenerate () {
        // Cache selection and contents in Quill's native format to be restored later
        this.quillDelta = this.editor.getContents();
        this.quillSelection = this.editor.getSelection();
        this.setState({
            generation: this.state.generation + 1
        });
    }

    focus () {
        this.editor.focus();
    }
    onEditorChangeText (value, delta, source, editor) {
        if (delta.ops !== this.getEditorContents()) {
            this.setState({ value: delta.ops }, () => {
                if (this.props.onChange) {
                    this.props.onChange(value, delta, source, editor);
                }
            });
        }
    }
    onEditorChangeSelection (range, source, editor) {
        const s = this.getEditorSelection() || {};
        const r = range || {};
        if (r.length !== s.length || r.index !== s.index) {
            this.setState({ selection: range });
            if (this.props.onChangeSelection) {
                this.props.onChangeSelection(range, source, editor);
            }
        }
    }
    onPaste (e) {
        const { onPaste } = this.props;
        if (onPaste) {
            onPaste.call(this, e);
        }
    }
    /*
     Renders an editor area, unless it has been provided one to clone.
     */
    renderEditingArea () {
        const self = this;
        const children = this.props.children;

        const properties = {
            key: this.state.generation,
            ref (element) {
                self.editingArea = element;
            }
        };

        const customElement = React.Children.count(children)
            ? React.Children.only(children)
            : null;

        const editingArea = customElement
            ? React.cloneElement(customElement, properties)
            : React.DOM.div(properties);

        return editingArea;
    }

    setEditorSelection (editor, r) {
        const range = r;
        if (range) {
            // Validate bounds before applying.
            const length = editor.getLength();
            range.index = Math.max(0, Math.min(range.index, length - 1));
            range.length = Math.max(0, Math.min(range.length, (length - 1) - range.index));
        }
        editor.setSelection(range);
    }

    setEditorContents (editor, value) {
        const delta = this.convertHtml(value);
        if (isEqual(delta, editor.getContents())) return;
        const sel = editor.getSelection();
        editor.setContents(delta || []);
        if (sel) this.setEditorSelection(editor, sel);
    }

    //创建编辑器
    createEditor ($el, config) {
        const editor = new Quill($el, config);
        this.hookEditor(editor);
        return editor;
    }

    hookEditor (editor) {
        // Expose the editor on change events via a weaker,
        // unprivileged proxy object that does not allow
        // accidentally modifying editor state.
        const unprivilegedEditor = this.makeUnprivilegedEditor(editor);

        this.handleTextChange = (delta, oldDelta, source) => {
            if (this.onEditorChangeText) {
                this.onEditorChangeText(
                    editor.root.innerHTML, delta, source,
                    unprivilegedEditor
                );
                this.onEditorChangeSelection(
                    editor.getSelection(), source,
                    unprivilegedEditor
                );
            }
        };

        this.handleSelectionChange = function (range, oldRange, source) {
            if (this.onEditorChangeSelection) {
                this.onEditorChangeSelection(
                    range, source,
                    unprivilegedEditor
                );
            }
        }.bind(this);
        this.handlePaste = function (e) {
            if (this.onPaste) {
                this.onPaste(e);
            }
        }.bind(this);
        editor.on('text-change', this.handleTextChange);
        editor.on('selection-change', this.handleSelectionChange);
        editor.root.addEventListener('paste', this.handlePaste);
    }

    unhookEditor (editor) {
        editor.off('selection-change');
        editor.off('editor-change');
    }

    makeUnprivilegedEditor (editor) {
        const {
            getLength,
            getText,
            getContents,
            getSelection,
            getBounds
        } = editor;
        return {
            getLength (...arg) { return getLength.apply(editor, arg); },
            getText (...arg) { return getText.apply(editor, arg); },
            getContents (...arg) { return getContents.apply(editor, arg); },
            getSelection (...arg) { return getSelection.apply(editor, arg); },
            getBounds (...arg) { return getBounds.apply(editor, arg); },
        };
    }

    render () {
        const {onKeyDown, onKeyPress, onKeyUp} = this.props;
        return (
            <div
                id={this.props.id}
                style={{position: 'relative'}}
                key={this.state.generation}
                onKeyPress={onKeyDown}
                onKeyDown={onKeyPress}
                onKeyUp={onKeyUp}
                className={['quill'].concat(this.props.className).join(' ')}
            >
                {this.renderEditingArea()}
                <div ref={target => this.pluginsTarget = target}/>
            </div>
        );
    }

}

QuillEditor.dirtyProps = [
    'modules',
    'formats',
    'bounds',
    'theme',
    'children',
    'plugins'
];

QuillEditor.cleanProps = [
    'id',
    'className',
    'style',
    'placeholder',
    'onKeyPress',
    'onKeyDown',
    'onKeyUp',
    'onChange',
    'onChangeSelection',
    'onPaste',
    'onSelectImage'
];



export default QuillEditor;