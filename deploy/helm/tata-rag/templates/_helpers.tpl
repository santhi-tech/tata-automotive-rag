{{/* Image, env and the shared image volume: identical for the API, UI and ingestion containers. */}}
{{- define "tata-rag.container" -}}
image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
imagePullPolicy: {{ .Values.image.pullPolicy }}
envFrom:
  - configMapRef:
      name: {{ .Release.Name }}-config
  - secretRef:
      name: {{ required "existingSecret is required (POSTGRES_USER, POSTGRES_PASSWORD, JWT_SECRET)" .Values.existingSecret }}
{{- end }}

{{- define "tata-rag.dataVolume" -}}
- name: data
  persistentVolumeClaim:
    claimName: {{ .Release.Name }}-data
{{- end }}
