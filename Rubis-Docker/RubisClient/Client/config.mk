##############################
#    Environment variables   #
##############################

JAVA  = $(JAVA_HOME)/bin/java
JAVAC = $(JAVA_HOME)/bin/javac
# +E -deprecation
JAVADOC = $(JAVA_HOME)/bin/javadoc
JAR = $(JAVA_HOME)/bin/jar
CLASSPATH=.

PARAM=all

MAKE = gmake
CP = /bin/cp
RM = /bin/rm
MKDIR = /bin/mkdir

##############################
#      awesome HELP          #
##############################


help: ## This help dialog
	@IFS=$$'\n' ; \
        printf "%35b %b\n" '\e[0;31m Rubis Client \e[0m' ;\
	help_lines=(`fgrep -h "##" $(where) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%-30s %s\n" "target" "help" ; \
	printf "%-30s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
	  IFS=$$':' ; \
          help_split=($$help_line) ; \
          help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
          help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
          printf '\033[36m'; \
          printf "%-30s %s" $$help_command ; \
          printf '\033[0m'; \
          printf "%s\n" $$help_info; \
	done


%.class: %.java
	${JAVAC} -classpath ${CLASSPATH} $<

