{{/* Common naming + label helpers. */}}

{{- define "latence.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "latence.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "latence.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
app.kubernetes.io/name: {{ include "latence.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "latence.selectorLabels" -}}
app.kubernetes.io/name: {{ include "latence.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* The in-cluster vLLM OCR Service name + base_url the pipeline's Parse stage targets. */}}
{{- define "latence.ocrServiceName" -}}
{{- printf "%s-vllm-ocr" (include "latence.fullname" .) -}}
{{- end -}}

{{- define "latence.ocrBaseUrl" -}}
{{- if .Values.stack.ocrBaseUrl -}}
{{- .Values.stack.ocrBaseUrl -}}
{{- else -}}
{{- printf "http://%s:%v/v1" (include "latence.ocrServiceName" .) .Values.vllmOcr.port -}}
{{- end -}}
{{- end -}}

{{- define "latence.llmServiceName" -}}
{{- printf "%s-vllm-llm" (include "latence.fullname" .) -}}
{{- end -}}

{{/* The env block shared by the pipeline pod: endpoint key (from the Secret) + metrics toggle. */}}
{{- define "latence.pipelineEnv" -}}
{{- $secretName := .Values.secrets.existingSecret -}}
{{- if and .Values.secrets.create (not $secretName) -}}
{{- $secretName = printf "%s-endpoint-keys" (include "latence.fullname" .) -}}
{{- end }}
{{- if $secretName }}
- name: OPENAI_API_KEY
  valueFrom:
    secretKeyRef:
      name: {{ $secretName }}
      key: OPENAI_API_KEY
      optional: true
{{- else }}
- name: OPENAI_API_KEY
  value: "EMPTY"
{{- end }}
{{- if ne .Values.metrics.kind "none" }}
- name: LATENCE_METRICS
  value: {{ .Values.metrics.kind | quote }}
- name: LATENCE_METRICS_TEXTFILE
  value: {{ .Values.metrics.textfilePath | quote }}
{{- end }}
{{- with .Values.pipeline.extraEnv }}
{{ toYaml . }}
{{- end }}
{{- end -}}
