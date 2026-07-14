{{/*
The shared pipeline pod template (used by both the Job and the CronJob). Renders the pod's
metadata + spec: the deploy/Dockerfile image running `latence run /etc/latence/stack.yaml`, the
stack ConfigMap mounted read-only, the data PVC mounted at /data, the endpoint-key env from the
Secret, and the non-root securityContext. Include with `nindent 6` under a Job's spec.template or a
CronJob's jobTemplate.spec.template.
*/}}
{{- define "latence.pipelinePodTemplate" -}}
metadata:
  labels:
    {{- include "latence.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: pipeline
spec:
  restartPolicy: {{ .Values.pipeline.restartPolicy }}
  securityContext:
    {{- toYaml .Values.podSecurityContext | nindent 4 }}
  {{- with .Values.imagePullSecrets }}
  imagePullSecrets:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with .Values.pipeline.nodeSelector }}
  nodeSelector:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with .Values.pipeline.tolerations }}
  tolerations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with .Values.pipeline.affinity }}
  affinity:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  containers:
    - name: pipeline
      image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
      imagePullPolicy: {{ .Values.image.pullPolicy }}
      args:
        - "run"
        - "/etc/latence/stack.yaml"
        - "--run-id"
        - {{ .Values.pipeline.runId | quote }}
      env:
        {{- include "latence.pipelineEnv" . | nindent 8 }}
      resources:
        {{- toYaml .Values.pipeline.resources | nindent 8 }}
      volumeMounts:
        - name: stack
          mountPath: /etc/latence
          readOnly: true
        - name: data
          mountPath: /data
  volumes:
    - name: stack
      configMap:
        name: {{ include "latence.fullname" . }}-stack
    - name: data
      {{- if .Values.persistence.data.enabled }}
      persistentVolumeClaim:
        claimName: {{ .Values.persistence.data.existingClaim | default (printf "%s-data" (include "latence.fullname" .)) }}
      {{- else }}
      emptyDir: {}
      {{- end }}
{{- end -}}
