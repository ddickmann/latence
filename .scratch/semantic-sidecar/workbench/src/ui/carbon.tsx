// The default controls: Carbon's own components, unstyled beyond Carbon. A host that wraps
// Carbon in its own components passes those instead (see INTEGRATION.md).
import { Button as CarbonButton, Checkbox as CarbonCheckbox, Select as CarbonSelect, SelectItem, TextArea as CarbonTextArea, TextInput } from "@carbon/react";

import type { ButtonProps, CheckboxProps, Controls, SelectProps, TextAreaProps, TextFieldProps } from "../host";

const KIND = { primary: "primary", secondary: "tertiary", quiet: "ghost", danger: "danger--tertiary" } as const;
const SIZE = { sm: "sm", md: "md", lg: "lg" } as const;

function Button({ kind = "secondary", size = "md", kbd, children, testId, disabled, onClick, ...aria }: ButtonProps) {
  return (
    <CarbonButton
      kind={KIND[kind]}
      size={SIZE[size]}
      className="rw-btn"
      type="button"
      disabled={disabled === true}
      data-testid={testId}
      {...(onClick === undefined ? {} : { onClick })}
      {...aria}
    >
      {children}
      {kbd !== undefined && <kbd className="rw-kbd" aria-hidden="true">{kbd}</kbd>}
    </CarbonButton>
  );
}

function TextField(props: TextFieldProps) {
  return (
    <TextInput
      id={props.id}
      labelText={props.label}
      size="md"
      value={props.value}
      placeholder={props.placeholder ?? ""}
      autoComplete="off"
      onChange={(e) => props.onChange(e.target.value)}
    />
  );
}

function TextArea(props: TextAreaProps) {
  return <CarbonTextArea id={props.id} labelText={props.label} rows={props.rows ?? 3} value={props.value} onChange={(e) => props.onChange(e.target.value)} />;
}

function Select(props: SelectProps) {
  return (
    <CarbonSelect id={props.id} labelText={props.label} size="md" value={props.value} onChange={(e) => props.onChange(e.target.value)}>
      {props.options.map((o) => <SelectItem key={o.value} value={o.value} text={o.text} />)}
    </CarbonSelect>
  );
}

function Checkbox(props: CheckboxProps) {
  return <CarbonCheckbox id={props.id} labelText={props.label} checked={props.checked} onChange={(_e, { checked }) => props.onChange(checked)} />;
}

export const carbonControls: Controls = { Button, TextField, TextArea, Select, Checkbox };
