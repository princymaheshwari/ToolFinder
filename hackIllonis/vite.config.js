export default {
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/detect': {
        target: 'http://10.83.217.96:8000',
        changeOrigin: true,
      }
    }
  }
}